"""SDFT with vLLM-accelerated generation.

Same algorithm as sdft.py but uses vLLM in colocate mode for ~5-10x faster
on-policy generation. The vLLM engine shares GPU memory with the training model
via sleep mode: vLLM yields memory during forward/backward, reclaims it for generation.

Requires: pip install vllm
Launch: torchrun --nproc_per_node=N src/selfdistillation/sdft_vllm.py --config configs/sdft_config.yaml
"""

import os
import sys
import yaml
import argparse
import random
import traceback
import copy
import time
import logging

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger("sdft_vllm")

import torch
import torch.nn.functional as F
from datasets import load_dataset, concatenate_datasets, load_from_disk
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainerCallback,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, TaskType


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def is_chat_format(x):
    return isinstance(x, list) and len(x) > 0 and isinstance(x[0], dict) and "role" in x[0]


def get_torch_dtype(train_cfg):
    if train_cfg.get("bf16") and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if train_cfg.get("fp16"):
        return torch.float16
    return torch.float32


# ── Data ─────────────────────────────────────────────────────────────────────


def load_split(path, split):
    p = Path(path)
    if p.exists() and (p / "dataset_dict.json").exists():
        return load_from_disk(path)[split]
    if p.exists() and (p / "dataset_info.json").exists():
        return load_from_disk(path)
    return load_dataset(path, split=split)


def format_dataset(ds, dcfg):
    """Normalize dataset to {query, demonstration} columns."""
    colmap = dcfg.get("column_map", {})
    for target, source in colmap.items():
        if source != target and source in ds.column_names and target not in ds.column_names:
            ds = ds.rename_column(source, target)

    if "query" in ds.column_names and "demonstration" in ds.column_names:
        return ds

    if "messages" in ds.column_names or "conversations" in ds.column_names:
        msg_col = "messages" if "messages" in ds.column_names else "conversations"

        def from_messages(ex):
            msgs = ex[msg_col]
            query_parts, demonstration = [], ""
            for i, m in enumerate(msgs):
                if m["role"] == "assistant" and i == len(msgs) - 1:
                    demonstration = m["content"]
                else:
                    if m["role"] == "user":
                        query_parts.append(m["content"])
                    elif m["role"] == "system":
                        query_parts.insert(0, m["content"])
            return {"query": "\n".join(query_parts), "demonstration": demonstration}

        return ds.map(from_messages, num_proc=8, remove_columns=ds.column_names)

    if "instruction" in ds.column_names:
        def from_instruction(ex):
            query = ex["instruction"]
            if ex.get("input"):
                query += "\n" + ex["input"]
            demonstration = ex.get("output") or ex.get("response", "")
            return {"query": query, "demonstration": demonstration}

        return ds.map(from_instruction, num_proc=8, remove_columns=ds.column_names)

    raise ValueError(
        f"Cannot detect SDFT format. Columns: {ds.column_names}. "
        "Expected 'query'+'demonstration', 'messages', 'conversations', or 'instruction'."
    )


def load_datasets(config):
    all_ds = []
    for dcfg in config.get("datasets", []):
        name, split = dcfg["name"], dcfg.get("split", "train")
        subset = dcfg.get("subset")

        print(f"Loading dataset: {name} (split={split})")
        ds = load_dataset(name, name=subset, split=split) if subset else load_split(name, split)

        max_samples = dcfg.get("max_samples")
        if max_samples and max_samples > 0:
            ds = ds.select(range(min(max_samples, len(ds))))

        ds = format_dataset(ds, dcfg)
        all_ds.append(ds)
        print(f"  -> {len(ds)} samples (columns: {ds.column_names})")

    combined = concatenate_datasets(all_ds) if len(all_ds) > 1 else all_ds[0]

    val_cfg = config.get("validation", {})
    split_ratio = val_cfg.get("split_ratio", 0.0)
    split_size = val_cfg.get("split_size")
    if split_ratio > 0 or split_size:
        val_size = min(split_size, len(combined)) if split_size else int(len(combined) * split_ratio)
        splits = combined.train_test_split(test_size=val_size, seed=val_cfg.get("seed", 42), shuffle=True)
        return splits["train"], splits["test"]

    return combined, None


# ── Model & Config ───────────────────────────────────────────────────────────


def load_tokenizer(config):
    tok = AutoTokenizer.from_pretrained(config["model"]["name"], trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(config):
    model_cfg = config["model"]
    train_cfg = config.get("training", {})
    model_name = model_cfg["name"]
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    torch_dtype = get_torch_dtype(train_cfg)
    is_ddp = int(os.environ.get("WORLD_SIZE", 1)) > 1

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    load_kwargs = dict(trust_remote_code=True, torch_dtype=torch_dtype)

    if model_cfg.get("load_in_4bit"):
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
        )
    if model_cfg.get("attn_implementation"):
        load_kwargs["attn_implementation"] = model_cfg["attn_implementation"]
    if model_cfg.get("rope_scaling"):
        load_kwargs["rope_scaling"] = model_cfg["rope_scaling"]
    if not is_ddp and torch.cuda.is_available():
        load_kwargs["device_map"] = {"": local_rank}

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    if is_ddp and torch.cuda.is_available():
        model.to(torch.device("cuda", local_rank))

    model.config.use_cache = False
    return model


def build_lora_config(config):
    lora_cfg = config.get("lora", {})
    if not lora_cfg.get("enabled"):
        return None
    return LoraConfig(
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("alpha", 16),
        lora_dropout=lora_cfg.get("dropout", 0.0),
        bias=lora_cfg.get("bias", "none"),
        task_type=TaskType.CAUSAL_LM,
        target_modules=lora_cfg.get("target_modules", [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]),
        use_rslora=lora_cfg.get("use_rslora", False),
    )


def build_training_args(config):
    train_cfg = config["training"]

    use_bf16 = bool(train_cfg.get("bf16"))
    use_fp16 = bool(train_cfg.get("fp16"))
    if use_bf16 and not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()):
        use_bf16, use_fp16 = False, True

    gc_kwargs = train_cfg.get("gradient_checkpointing_kwargs")
    if train_cfg.get("gradient_checkpointing") and gc_kwargs is None:
        gc_kwargs = {"use_reentrant": False}

    return TrainingArguments(
        output_dir=train_cfg.get("output_dir", "./outputs/sdft"),
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 1),
        per_device_eval_batch_size=train_cfg.get("per_device_eval_batch_size", 1),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 1),
        num_train_epochs=train_cfg.get("num_train_epochs", 1),
        max_steps=train_cfg.get("max_steps", -1),
        learning_rate=float(train_cfg.get("learning_rate", 5e-6)),
        warmup_steps=train_cfg.get("warmup_steps", 0),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        weight_decay=train_cfg.get("weight_decay", 0.0),
        logging_steps=train_cfg.get("logging_steps", 1),
        save_strategy=train_cfg.get("save_strategy", "steps"),
        save_steps=train_cfg.get("save_steps", 500),
        report_to=train_cfg.get("report_to", "none"),
        fp16=use_fp16,
        bf16=use_bf16,
        eval_strategy=train_cfg.get("eval_strategy", "no"),
        eval_steps=train_cfg.get("eval_steps", 200),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", False),
        gradient_checkpointing_kwargs=gc_kwargs,
        optim=train_cfg.get("optim", "adamw_torch"),
        max_grad_norm=train_cfg.get("max_grad_norm", 1.0),
        ddp_find_unused_parameters=train_cfg.get("ddp_find_unused_parameters", False),
        remove_unused_columns=False,
    )


# ── vLLM Engine ──────────────────────────────────────────────────────────────


def init_vllm_engine(config):
    """Initialize vLLM LLM engine in colocate mode with sleep support."""
    from vllm import LLM

    model_name = config["model"]["name"]
    vllm_cfg = config.get("vllm", {})

    llm = LLM(
        model=model_name,
        tensor_parallel_size=vllm_cfg.get("tensor_parallel_size", 1),
        gpu_memory_utilization=vllm_cfg.get("gpu_memory_utilization", 0.3),
        dtype="bfloat16" if config.get("training", {}).get("bf16") else "auto",
        trust_remote_code=True,
        distributed_executor_backend="external_launcher",
        enable_sleep_mode=vllm_cfg.get("enable_sleep_mode", True),
    )
    logger.info("[vllm] Engine initialized (gpu_mem=%.0f%%)",
                vllm_cfg.get("gpu_memory_utilization", 0.3) * 100)
    return llm


def sync_vllm_weights(llm, model):
    """Copy training model weights into vLLM engine."""
    from vllm.worker.model_runner import ModelRunner

    unwrapped = model
    while hasattr(unwrapped, "module"):
        unwrapped = unwrapped.module

    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    src_state = unwrapped.state_dict()
    llm_model.load_weights(src_state.items())
    logger.info("[vllm] Weights synced from training model")


# ── SDFT Trainer (vLLM) ─────────────────────────────────────────────────────


class SDFTVLLMTrainer(Trainer):
    """SDFT trainer with vLLM-accelerated generation.

    Uses vLLM in colocate mode with sleep: during generation vLLM wakes up and
    uses GPU memory, during forward/backward it sleeps and yields memory back.
    """

    def __init__(self, tokenizer, sdft_config, vllm_engine, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer
        self.sdft_config = sdft_config
        self.vllm_engine = vllm_engine

        # EMA teacher: deep copy of the student, frozen
        unwrapped = self._unwrap_model(self.model)
        self.ema_model = copy.deepcopy(unwrapped)
        self.ema_model.requires_grad_(False)
        self.ema_model.eval()

        self.ema_alpha = sdft_config.get("ema_alpha", 0.02)
        self.max_gen_length = sdft_config.get("max_gen_length", 2048)
        self.gen_temperature = sdft_config.get("generation_temperature", 1.0)
        self.gen_top_p = sdft_config.get("generation_top_p")
        self.gen_top_k = sdft_config.get("generation_top_k")
        self.mask_first_n = sdft_config.get("mask_first_n_tokens", 0)
        self.teacher_prompt_template = sdft_config.get("teacher_prompt_template", (
            "{query}\n\n"
            "This is an example for a response to the question:\n"
            "{demonstration}\n\n"
            "Now answer with a response of your own, including the thinking process:\n"
        ))

        self._steps_since_sync = 0
        self._sync_interval = sdft_config.get("vllm_sync_interval", 1)

    @staticmethod
    def _unwrap_model(model):
        while hasattr(model, "module"):
            model = model.module
        return model

    def _build_student_context(self, query):
        """Build student input: chat template with query only."""
        messages = [{"role": "user", "content": query}]
        out = self.tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        )
        if isinstance(out, torch.Tensor):
            return out[0]
        return out["input_ids"][0]

    def _build_teacher_context(self, query, demonstration):
        """Build teacher input: chat template with query + demonstration in-context."""
        teacher_content = self.teacher_prompt_template.format(
            query=query, demonstration=demonstration
        )
        messages = [{"role": "user", "content": teacher_content}]
        out = self.tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        )
        if isinstance(out, torch.Tensor):
            return out[0]
        return out["input_ids"][0]

    def _build_student_prompts_text(self, queries):
        """Build tokenized student prompts as text for vLLM."""
        prompts = []
        for query in queries:
            messages = [{"role": "user", "content": query}]
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompts.append(text)
        return prompts

    def _generate_student_rollouts_vllm(self, queries, device):
        """Generate responses using vLLM engine (much faster than model.generate)."""
        from vllm import SamplingParams

        # Sync weights to vLLM if needed
        if self._steps_since_sync >= self._sync_interval:
            sync_vllm_weights(self.vllm_engine, self.model)
            self._steps_since_sync = 0

        # Wake up vLLM engine
        self.vllm_engine.wake_up()

        sampling_params = SamplingParams(
            max_tokens=self.max_gen_length,
            temperature=self.gen_temperature,
            top_p=self.gen_top_p if self.gen_top_p is not None else 1.0,
            top_k=self.gen_top_k if self.gen_top_k is not None else -1,
        )

        prompts = self._build_student_prompts_text(queries)

        t0 = time.time()
        logger.info("[vllm-gen] Generating for %d queries (max_tokens=%d)", len(queries), self.max_gen_length)

        outputs = self.vllm_engine.generate(prompts, sampling_params=sampling_params, use_tqdm=False)

        # Put vLLM to sleep to free GPU memory for training
        self.vllm_engine.sleep()

        all_prompt_ids = []
        all_gen_ids = []

        for qi, output in enumerate(outputs):
            prompt_ids = self._build_student_context(queries[qi]).to(device)
            gen_token_ids = output.outputs[0].token_ids
            gen_ids = torch.tensor(gen_token_ids, dtype=torch.long, device=device)

            elapsed = time.time() - t0
            logger.info("[vllm-gen] Query %d/%d: generated %d tokens (%.1f tok/s total)",
                        qi + 1, len(queries), gen_ids.shape[0],
                        sum(len(o.outputs[0].token_ids) for o in outputs) / max(elapsed, 0.01))

            all_prompt_ids.append(prompt_ids)
            all_gen_ids.append(gen_ids)

        elapsed = time.time() - t0
        total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        logger.info("[vllm-gen] Done: %d tokens in %.1fs (%.1f tok/s)",
                    total_tokens, elapsed, total_tokens / max(elapsed, 0.01))

        return all_prompt_ids, all_gen_ids

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        device = next(model.parameters()).device
        queries = inputs["query"]
        demonstrations = inputs["demonstration"]
        logger.info("[loss] compute_loss called, batch_size=%d", len(queries))

        # Step 1: Generate y ~ π_θ(·|x) using vLLM
        t0 = time.time()
        student_prompt_ids, gen_ids_list = self._generate_student_rollouts_vllm(queries, device)
        logger.info("[loss] Generation done in %.1fs", time.time() - t0)

        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        total_tokens = 0

        for i in range(len(queries)):
            gen_ids = gen_ids_list[i]
            if gen_ids.numel() == 0:
                continue

            # Step 2: Build student input = student_context + generated tokens
            s_prompt = student_prompt_ids[i]
            student_input_ids = torch.cat([s_prompt, gen_ids], dim=0).unsqueeze(0)

            # Step 3: Build teacher input = teacher_context + generated tokens
            t_prompt = self._build_teacher_context(queries[i], demonstrations[i]).to(device)
            teacher_input_ids = torch.cat([t_prompt, gen_ids], dim=0).unsqueeze(0)

            # Create attention masks
            student_attn = torch.ones_like(student_input_ids)
            teacher_attn = torch.ones_like(teacher_input_ids)

            # Step 4: Student forward pass (with grad)
            student_outputs = model(input_ids=student_input_ids, attention_mask=student_attn)
            s_start = s_prompt.shape[0] - 1
            s_end = s_start + gen_ids.shape[0]
            student_logits = student_outputs.logits[0, s_start:s_end]

            # Step 5: Teacher forward pass (no grad, EMA model)
            with torch.no_grad():
                teacher_outputs = self.ema_model(input_ids=teacher_input_ids, attention_mask=teacher_attn)
                t_start = t_prompt.shape[0] - 1
                t_end = t_start + gen_ids.shape[0]
                teacher_logits = teacher_outputs.logits[0, t_start:t_end]

            # Step 6: KL divergence — reverse KL: KL(π_θ || π_φ)
            student_logprobs = F.log_softmax(student_logits, dim=-1)
            teacher_logprobs = F.log_softmax(teacher_logits, dim=-1)

            gen_len = gen_ids.shape[0]
            gen_mask = torch.ones(gen_len, device=device)
            if self.mask_first_n > 0:
                mask_len = min(self.mask_first_n, gen_len)
                gen_mask[:mask_len] = 0.0

            per_token_kl = F.kl_div(
                teacher_logprobs, student_logprobs, log_target=True, reduction="none"
            ).sum(dim=-1)

            masked_kl = per_token_kl * gen_mask
            n_tokens = gen_mask.sum().clamp(min=1)
            total_loss = total_loss + masked_kl.sum() / n_tokens
            total_tokens += n_tokens.item()

        batch_size = len(queries)
        loss = total_loss / max(batch_size, 1)
        logger.info("[loss] KL loss=%.4f, total_tokens=%d", loss.item(), total_tokens)

        if return_outputs:
            return loss, {"loss": loss.detach()}
        return loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)
        self._update_ema()
        self._steps_since_sync += 1
        return loss

    def _update_ema(self):
        """EMA update: φ ← (1-α)·φ + α·θ"""
        unwrapped = self._unwrap_model(self.model)
        alpha = self.ema_alpha
        with torch.no_grad():
            for ema_param, student_param in zip(self.ema_model.parameters(), unwrapped.parameters()):
                ema_param.data.mul_(1.0 - alpha).add_(student_param.data, alpha=alpha)

    def get_train_dataloader(self):
        """Override to use a simple dataloader that passes raw dicts."""
        from torch.utils.data import DataLoader

        def collate_fn(batch):
            return {
                "query": [b["query"] for b in batch],
                "demonstration": [b["demonstration"] for b in batch],
            }

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0,
            drop_last=True,
        )

    def get_eval_dataloader(self, eval_dataset=None):
        """Override to use same collation for eval."""
        from torch.utils.data import DataLoader

        ds = eval_dataset if eval_dataset is not None else self.eval_dataset

        def collate_fn(batch):
            return {
                "query": [b["query"] for b in batch],
                "demonstration": [b["demonstration"] for b in batch],
            }

        return DataLoader(
            ds,
            batch_size=self.args.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
        )


# ── Eval Callback ────────────────────────────────────────────────────────────


class EvalSampleLoggerCallback(TrainerCallback):
    """Generate and log student model outputs on fixed eval queries at each eval step."""

    def __init__(self, tokenizer, eval_dataset, vllm_engine, num_samples=5, max_new_tokens=256, use_wandb=False):
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.use_wandb = use_wandb
        self.vllm_engine = vllm_engine
        n = min(num_samples, len(eval_dataset))
        self.samples = [eval_dataset[i] for i in random.sample(range(len(eval_dataset)), n)]

    @staticmethod
    def _unwrap(model):
        while hasattr(model, "module"):
            model = model.module
        return model

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if model is None:
            return

        is_main = int(os.environ.get("RANK", 0)) == 0

        from vllm import SamplingParams

        self.vllm_engine.wake_up()

        prompts = []
        for sample in self.samples:
            messages = [{"role": "user", "content": sample["query"]}]
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts.append(text)

        sampling_params = SamplingParams(
            max_tokens=self.max_new_tokens,
            temperature=0.0,
        )
        outputs = self.vllm_engine.generate(prompts, sampling_params=sampling_params, use_tqdm=False)

        self.vllm_engine.sleep()

        rows = []
        for sample, output in zip(self.samples, outputs):
            generated = output.outputs[0].text[:500]
            row = {
                "query": sample["query"][:500],
                "demonstration": sample["demonstration"][:500],
                "generated": generated,
            }
            rows.append(row)

            if is_main:
                print(f"\n{'='*80}")
                print(f"[Eval sample - step {state.global_step}]")
                for k in ("query", "demonstration", "generated"):
                    print(f"{k.upper():15s} {row[k][:300]}")
                print("=" * 80)

        if is_main and self.use_wandb:
            try:
                import wandb
                if wandb.run:
                    table = wandb.Table(
                        columns=["query", "demonstration", "generated"],
                        data=[[r[c] for c in ("query", "demonstration", "generated")] for r in rows],
                    )
                    wandb.log({"eval_samples": table, "global_step": state.global_step})
            except Exception:
                pass


# ── Training ─────────────────────────────────────────────────────────────────


def setup_wandb(config):
    wandb_cfg = config.get("wandb", {})
    if not wandb_cfg.get("enabled") or int(os.environ.get("RANK", 0)) != 0:
        return
    import wandb
    wandb.init(
        project=wandb_cfg.get("project", "sdft-training"),
        entity=wandb_cfg.get("entity"),
        name=wandb_cfg.get("run_name"),
        tags=wandb_cfg.get("tags", []),
        config=config,
    )


def train(config):
    print(f"[Rank {os.environ.get('RANK', 0)}] Starting SDFT training (vLLM)")

    setup_wandb(config)
    tokenizer = load_tokenizer(config)
    train_ds, eval_ds = load_datasets(config)

    # Init vLLM engine first (before training model to manage GPU memory)
    vllm_engine = init_vllm_engine(config)

    model = load_model(config)
    training_args = build_training_args(config)
    peft_config = build_lora_config(config)

    if peft_config is not None:
        from peft import get_peft_model
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    sdft_cfg = config.get("sdft", {})

    # Initial weight sync
    sync_vllm_weights(vllm_engine, model)
    vllm_engine.sleep()

    callbacks = []
    eval_samples_cfg = config.get("eval_samples", {})
    if eval_ds and eval_samples_cfg.get("enabled", True):
        callbacks.append(EvalSampleLoggerCallback(
            tokenizer=tokenizer,
            eval_dataset=eval_ds,
            vllm_engine=vllm_engine,
            num_samples=eval_samples_cfg.get("num_samples", 5),
            max_new_tokens=eval_samples_cfg.get("max_new_tokens", 256),
            use_wandb=config.get("wandb", {}).get("enabled", False),
        ))

    trainer = SDFTVLLMTrainer(
        tokenizer=tokenizer,
        sdft_config=sdft_cfg,
        vllm_engine=vllm_engine,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        callbacks=callbacks or None,
    )

    resume_from = config.get("training", {}).get("resume_from_checkpoint")
    trainer.train(resume_from_checkpoint=resume_from)

    outdir = config["training"].get("output_dir", "./outputs/sdft")
    trainer.save_model(outdir)
    tokenizer.save_pretrained(outdir)
    print("SDFT training complete.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
