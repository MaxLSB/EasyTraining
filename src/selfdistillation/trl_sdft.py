"""On-policy Self-Distillation Fine-Tuning (SDFT)"""

import os

# vLLM V1 defaults to multiprocessing workers, which prevents passing tensors
# through collective_rpc (they'd need to be serialized across processes).
# For single-GPU SDFT we keep the worker in-process so weight syncing is a
# direct in-memory copy.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

import copy
import yaml
import argparse

import torch
import torch.nn.functional as F
from datasets import load_dataset, concatenate_datasets, load_from_disk
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler
from vllm import LLM, SamplingParams
from torch.utils.data import DataLoader
from trl.trainer.utils import disable_dropout_in_model


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def get_torch_dtype(train_cfg):
    if (
        train_cfg.get("bf16")
        and torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
    ):
        return torch.bfloat16
    if train_cfg.get("fp16"):
        return torch.float16
    return torch.float32


def _get_model_name(config):
    """Resolve model name from config (supports 'name' or legacy 'student' key)."""
    model_cfg = config["model"]
    return model_cfg.get("name", model_cfg.get("student"))


# ── Data ─────────────────────────────────────────────────────────────────────


def load_split(path, split):
    p = Path(path)
    if p.exists() and (p / "dataset_dict.json").exists():
        return load_from_disk(path)[split]
    if p.exists() and (p / "dataset_info.json").exists():
        return load_from_disk(path)
    return load_dataset(path, split=split)


def load_datasets(config):
    """Load datasets keeping both 'prompt' and 'completion' columns.

    D = {(xi, ci)} where xi = prompt, ci = demonstration.
    """
    all_ds = []
    for dcfg in config.get("datasets", []):
        name, split = dcfg["name"], dcfg.get("split", "train")
        subset = dcfg.get("subset")

        print(f"Loading dataset: {name} (split={split})")
        ds = (
            load_dataset(name, name=subset, split=split)
            if subset
            else load_split(name, split)
        )

        max_samples = dcfg.get("max_samples")
        if max_samples and max_samples > 0:
            ds = ds.select(range(min(max_samples, len(ds))))

        col_map = dcfg.get("column_map", {})
        prompt_src = col_map.get("prompt", "prompt")
        completion_src = col_map.get("completion", "completion")

        if prompt_src != "prompt" and prompt_src in ds.column_names:
            ds = ds.rename_column(prompt_src, "prompt")
        if completion_src != "completion" and completion_src in ds.column_names:
            ds = ds.rename_column(completion_src, "completion")

        keep_cols = {"prompt", "completion"} & set(ds.column_names)
        ds = ds.remove_columns([c for c in ds.column_names if c not in keep_cols])
        all_ds.append(ds)
        print(f"  -> {len(ds)} samples (columns: {ds.column_names})")

    combined = concatenate_datasets(all_ds) if len(all_ds) > 1 else all_ds[0]

    val_cfg = config.get("validation", {})
    split_ratio = val_cfg.get("split_ratio", 0.0)
    split_size = val_cfg.get("split_size")
    if split_ratio > 0 or split_size:
        val_size = (
            min(split_size, len(combined))
            if split_size
            else int(len(combined) * split_ratio)
        )
        splits = combined.train_test_split(
            test_size=val_size, seed=val_cfg.get("seed", 42), shuffle=True
        )
        return splits["train"], splits["test"]

    return combined, None


def filter_long_prompts(
    dataset,
    tokenizer,
    max_prompt_length,
    student_system_prompt=None,
    teacher_system_prompt=None,
):
    """Remove samples where student or teacher formatted prompt exceeds max_prompt_length."""
    initial_len = len(dataset)

    def _tokenize_len(text):
        return len(tokenizer.encode(text, add_special_tokens=False))

    def _fits(example):
        # Student prompt: chat-templated user prompt
        s_messages = []
        if student_system_prompt:
            s_messages.append({"role": "system", "content": student_system_prompt})
        s_messages.append({"role": "user", "content": example["prompt"]})
        s_text = tokenizer.apply_chat_template(
            s_messages, tokenize=False, add_generation_prompt=True
        )

        # Teacher prompt: prompt + demonstration + template
        teacher_content = (
            f"{example['prompt']}\n"
            f"This is an example for a response to the question:\n"
            f"{example['completion']}\n"
            f"Now answer with a response of your own, including the thinking process:"
        )
        t_messages = []
        if teacher_system_prompt:
            t_messages.append({"role": "system", "content": teacher_system_prompt})
        t_messages.append({"role": "user", "content": teacher_content})
        t_text = tokenizer.apply_chat_template(
            t_messages, tokenize=False, add_generation_prompt=True
        )

        return (
            _tokenize_len(s_text) <= max_prompt_length
            and _tokenize_len(t_text) <= max_prompt_length
        )

    filtered = dataset.filter(_fits, num_proc=4)
    removed = initial_len - len(filtered)
    print(
        f"  Filtered out {removed}/{initial_len} samples exceeding "
        f"{max_prompt_length} prompt tokens ({len(filtered)} remaining)"
    )
    return filtered


# ── Model & vLLM ─────────────────────────────────────────────────────────────


def load_tokenizer(config):
    model_cfg = config["model"]
    tok = AutoTokenizer.from_pretrained(
        model_cfg.get("tokenizer", _get_model_name(config)), trust_remote_code=True
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok


def load_student_model(config):
    model_cfg = config["model"]
    train_cfg = config.get("training", {})
    torch_dtype = get_torch_dtype(train_cfg)

    model = AutoModelForCausalLM.from_pretrained(
        _get_model_name(config),
        torch_dtype=torch_dtype,
        attn_implementation=model_cfg.get("attn_implementation", "flash_attention_2"),
        trust_remote_code=True,
        device_map={"": 0},
    )
    model.config.use_cache = False

    if train_cfg.get("gradient_checkpointing"):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    return model


def init_teacher_from_student(student):
    """Algorithm 1, line 1: Set teacher weights ϕ = θ."""
    teacher = copy.deepcopy(student)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher


def setup_vllm(config):
    """Set up vLLM with the *student* model for on-policy generation."""
    sdft_cfg = config.get("sdft", {})
    max_prompt = sdft_cfg.get("max_prompt_length", 1024)
    max_completion = sdft_cfg.get("max_completion_length", 1024)

    llm = LLM(
        model=_get_model_name(config),
        gpu_memory_utilization=sdft_cfg.get("vllm_gpu_memory_utilization", 0.4),
        max_model_len=max_prompt + max_completion,
        enable_sleep_mode=True,
        dtype="auto",
        trust_remote_code=True,
        max_num_batched_tokens=4096,
    )
    llm.sleep(level=1)
    return llm


# ── Helpers ──────────────────────────────────────────────────────────────────


def ema_update(teacher, student, alpha):
    """Algorithm 1, line 19: ϕ ← α·θ + (1 − α)·ϕ"""
    with torch.no_grad():
        for t_param, s_param in zip(teacher.parameters(), student.parameters()):
            t_param.data.mul_(1 - alpha).add_(s_param.data, alpha=alpha)


def _load_weights_on_worker(worker, weights):
    """collective_rpc callback: load weights on a single vLLM worker."""
    worker.model_runner.model.load_weights(weights)


def sync_vllm_weights(llm, model):
    """Push current student weights into the vLLM engine (V1 compatible)."""
    weights = [(name, param.data) for name, param in model.named_parameters()]
    llm.collective_rpc(_load_weights_on_worker, args=(weights,))


def format_student_prompts(prompts, tokenizer, system_prompt=None):
    """CtxS(x): Student context – prompt only."""
    formatted = []
    for prompt in prompts:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        formatted.append(text)
    return formatted


def format_teacher_prompts(prompts, completions, tokenizer, system_prompt=None):
    """CtxT(x, c): Teacher context – prompt enriched with the demonstration c.

    Paper template (Appendix B):
        <Question>
        This is an example for a response to the question:
        <Demonstration>
        Now answer with a response of your own, including the thinking process:
    """
    formatted = []
    for prompt, completion in zip(prompts, completions):
        teacher_content = (
            f"{prompt}\n"
            f"This is an example for a response to the question:\n"
            f"{completion}\n"
            f"Now answer with a response of your own, including the thinking process:"
        )
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": teacher_content})
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        formatted.append(text)
    return formatted


def generate_completions(llm, student, prompts_text, sdft_cfg):
    """Sync student weights → wake vLLM → on-policy rollout → sleep."""
    sampling_params = SamplingParams(
        temperature=sdft_cfg.get("temperature", 0.7),
        top_p=sdft_cfg.get("top_p", 0.9),
        top_k=sdft_cfg.get("top_k", -1),
        min_p=sdft_cfg.get("min_p", 0.0),
        max_tokens=sdft_cfg.get("max_completion_length", 1024),
        repetition_penalty=sdft_cfg.get("repetition_penalty", 1.0),
    )

    torch.cuda.empty_cache()
    llm.wake_up()
    sync_vllm_weights(llm, student)
    outputs = llm.generate(
        prompts_text, sampling_params=sampling_params, use_tqdm=False
    )
    llm.sleep(level=1)
    torch.cuda.empty_cache()

    return [list(output.outputs[0].token_ids) for output in outputs]


def prepare_inputs(
    prompt_texts, completion_ids_list, tokenizer, max_prompt_length, device
):
    """Tokenize prompts (left-padded) + pad completions (right-padded).

    Returns:
        input_ids:       (B, P+C)
        attention_mask:  (B, P+C)
        completion_mask: (B, C)  mask for valid completion tokens
        logits_to_keep:  int
    """
    prompt_enc = tokenizer(
        prompt_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_prompt_length,
        add_special_tokens=False,
    )
    prompt_ids = prompt_enc["input_ids"].to(device)
    prompt_mask = prompt_enc["attention_mask"].to(device)

    max_comp_len = max(len(ids) for ids in completion_ids_list)
    padded_comp_ids, comp_masks = [], []
    for ids in completion_ids_list:
        pad_len = max_comp_len - len(ids)
        padded_comp_ids.append(ids + [tokenizer.pad_token_id] * pad_len)
        comp_masks.append([1] * len(ids) + [0] * pad_len)

    completion_ids = torch.tensor(padded_comp_ids, dtype=torch.long, device=device)
    completion_mask = torch.tensor(comp_masks, dtype=torch.long, device=device)

    input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
    return input_ids, attention_mask, completion_mask, max_comp_len


def get_completion_logits(model, input_ids, attention_mask, logits_to_keep):
    """Forward pass → return logits for the completion positions only."""
    logits = model(
        input_ids=input_ids, attention_mask=attention_mask, use_cache=False
    ).logits
    # logits[:, i] predicts input_ids[:, i+1]; drop the last position
    logits = logits[:, :-1, :]
    # keep only the last `logits_to_keep` positions (= completion tokens)
    return logits[:, -logits_to_keep:, :]


# ── Loss ─────────────────────────────────────────────────────────────────────


def reverse_kl_loss(
    student_logits, teacher_logits, completion_mask, temperature=1.0, mask_first_n=0
):
    """Reverse KL on full vocab (Eq. A.1): KL(student || teacher).

    F.kl_div(input=log_p, target=log_q, log_target=True) = Σ q·(log q − log p) = KL(q‖p)
    Here q = student, p = teacher → KL(student ‖ teacher).

    mask_first_n: zero-out loss on the first N completion tokens to suppress
    spurious linguistic patterns from the demonstration (paper §5 discussion).
    """
    if mask_first_n > 0:
        completion_mask = completion_mask.clone()
        completion_mask[:, :mask_first_n] = 0

    student_lp = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_lp = F.log_softmax(teacher_logits / temperature, dim=-1)

    per_token_kl = F.kl_div(
        teacher_lp, student_lp, log_target=True, reduction="none"
    ).sum(
        dim=-1
    )  # (B, C)

    return (per_token_kl * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)


# ── Training ─────────────────────────────────────────────────────────────────


def setup_wandb(config):
    wandb_cfg = config.get("wandb", {})
    if not wandb_cfg.get("enabled"):
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
    print("Starting Self-Distillation Fine-Tuning (SDFT) – Algorithm 1")

    train_cfg = config.get("training", {})
    sdft_cfg = config.get("sdft", {})
    device = torch.device("cuda")

    setup_wandb(config)
    tokenizer = load_tokenizer(config)
    train_ds, _eval_ds = load_datasets(config)

    max_prompt_length = sdft_cfg.get("max_prompt_length", 1024)
    student_system = sdft_cfg.get("student_system_prompt")
    teacher_system = sdft_cfg.get("teacher_system_prompt")
    train_ds = filter_long_prompts(
        train_ds, tokenizer, max_prompt_length, student_system, teacher_system
    )

    # ── Algorithm 1, line 1: ϕ = θ ────────────────────────────────────────────
    print("Loading student model (πθ)...")
    student = load_student_model(config)
    print("Initializing teacher from student (EMA copy, πϕ)...")
    teacher = init_teacher_from_student(student)

    if sdft_cfg.get("disable_dropout", True):
        disable_dropout_in_model(student)
        disable_dropout_in_model(teacher)

    print("Setting up vLLM for on-policy student generation...")
    llm = setup_vllm(config)

    # Hyper-parameters
    batch_size = train_cfg.get("per_device_train_batch_size", 2)
    grad_accum = train_cfg.get("gradient_accumulation_steps", 1)
    num_epochs = train_cfg.get("num_train_epochs", 1)
    lr = float(train_cfg.get("learning_rate", 1e-5))
    kl_temperature = sdft_cfg.get("kl_temperature", 1.0)
    ema_alpha = sdft_cfg.get("ema_alpha", 0.02)
    mask_first_n = sdft_cfg.get("mask_first_n_tokens", 0)
    steps_per_gen = sdft_cfg.get("steps_per_generation", 4)
    gen_batch_size = batch_size * steps_per_gen

    # DataLoader – yields D = {(xi, ci)} batches
    dataloader = DataLoader(
        train_ds,
        batch_size=gen_batch_size,
        shuffle=True,
        collate_fn=lambda batch: {
            "prompts": [x["prompt"] for x in batch],
            "completions": [x["completion"] for x in batch],
        },
        drop_last=True,
    )

    # Optimizer & scheduler
    total_steps = (len(dataloader) * steps_per_gen * num_epochs) // grad_accum
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=lr,
        weight_decay=train_cfg.get("weight_decay", 0.01),
    )
    scheduler = get_scheduler(
        train_cfg.get("lr_scheduler_type", "cosine"),
        optimizer=optimizer,
        num_warmup_steps=train_cfg.get("warmup_steps", 0),
        num_training_steps=total_steps,
    )

    # AMP (GradScaler only needed for fp16, not bf16)
    torch_dtype = get_torch_dtype(train_cfg)
    use_amp = torch_dtype != torch.float32
    use_scaler = torch_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    # Training loop
    global_step = 0
    micro_step = 0
    accum_loss = 0.0
    student.train()
    output_dir = train_cfg.get("output_dir", "./outputs/sdft")
    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(num_epochs):
        for batch in dataloader:
            all_prompts = batch["prompts"]
            all_completions = batch["completions"]

            # ── Lines 5-7: On-policy student rollout via vLLM ─────────────
            # CtxS(x): student context (prompt only)
            student_prompts_all = format_student_prompts(
                all_prompts, tokenizer, student_system
            )
            # Sample yi ~ Psample(· | si) using student weights
            comp_ids_all = generate_completions(
                llm, student, student_prompts_all, sdft_cfg
            )

            # Split into training sub-batches
            prompt_batches = [
                all_prompts[i : i + batch_size]
                for i in range(0, len(all_prompts), batch_size)
            ]
            completion_batches = [
                all_completions[i : i + batch_size]
                for i in range(0, len(all_completions), batch_size)
            ]
            comp_id_batches = [
                comp_ids_all[i : i + batch_size]
                for i in range(0, len(comp_ids_all), batch_size)
            ]

            # ── Lines 8-19: Training phase ────────────────────────────────
            for prompts, completions, comp_ids_list in zip(
                prompt_batches, completion_batches, comp_id_batches
            ):
                # Line 6: si ← CtxS(xi)
                s_texts = format_student_prompts(prompts, tokenizer, student_system)
                # Line 9: ti ← CtxT(xi, ci)
                t_texts = format_teacher_prompts(
                    prompts, completions, tokenizer, teacher_system
                )

                s_ids, s_mask, c_mask, ltk = prepare_inputs(
                    s_texts, comp_ids_list, tokenizer, max_prompt_length, device
                )
                t_ids, t_mask, _, _ = prepare_inputs(
                    t_texts, comp_ids_list, tokenizer, max_prompt_length, device
                )

                with torch.amp.autocast("cuda", dtype=torch_dtype, enabled=use_amp):
                    # Line 11: l^S_{i,t} ← log πθ(yi,t | yi,<t, si)
                    student_logits = get_completion_logits(student, s_ids, s_mask, ltk)
                    # Line 12: l^T_{i,t} ← log πϕ(yi,t | yi,<t, ti)
                    with torch.no_grad():
                        teacher_logits = get_completion_logits(
                            teacher, t_ids, t_mask, ltk
                        )

                    # Lines 15-16: gradient via reverse KL (Eq. A.1)
                    loss = reverse_kl_loss(
                        student_logits,
                        teacher_logits,
                        c_mask,
                        kl_temperature,
                        mask_first_n,
                    )
                    loss = loss / grad_accum

                scaler.scale(loss).backward()
                accum_loss += loss.item()
                micro_step += 1

                if micro_step % grad_accum == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        student.parameters(), train_cfg.get("max_grad_norm", 1.0)
                    )
                    # Line 18: θ ← θ − η g
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    # Line 19: ϕ ← α·θ + (1 − α)·ϕ
                    ema_update(teacher, student, ema_alpha)

                    if global_step % train_cfg.get("logging_steps", 1) == 0:
                        cur_lr = scheduler.get_last_lr()[0]
                        print(
                            f"[Epoch {epoch+1}/{num_epochs}] "
                            f"Step {global_step}/{total_steps} | "
                            f"loss={accum_loss:.4f} | lr={cur_lr:.2e}"
                        )
                        try:
                            import wandb

                            if wandb.run:
                                wandb.log(
                                    {
                                        "train/loss": accum_loss,
                                        "train/lr": cur_lr,
                                        "train/epoch": epoch,
                                        "train/global_step": global_step,
                                    },
                                    step=global_step,
                                )
                        except ImportError:
                            pass
                        accum_loss = 0.0

                    save_steps = train_cfg.get("save_steps")
                    if save_steps and global_step % save_steps == 0:
                        ckpt = os.path.join(output_dir, f"checkpoint-{global_step}")
                        student.save_pretrained(ckpt)
                        tokenizer.save_pretrained(ckpt)
                        print(f"  Saved checkpoint to {ckpt}")

    student.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Training complete. Model saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
