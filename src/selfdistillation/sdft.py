"""Self-Distillation Fine-Tuning (SDFT) using DistilTrainer."""

import os
import sys
import yaml
import argparse

import torch
from datasets import load_dataset, concatenate_datasets, load_from_disk
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

# Ensure local imports (sdft_trainer, sdft_config) are available
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sdft_trainer import DistilTrainer
from sdft_config import DistilConfig


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


# ── Data ─────────────────────────────────────────────────────────────────────


def load_split(path, split):
    p = Path(path)
    if p.exists() and (p / "dataset_dict.json").exists():
        return load_from_disk(path)[split]
    if p.exists() and (p / "dataset_info.json").exists():
        return load_from_disk(path)
    return load_dataset(path, split=split)


def load_datasets(config):
    """Load datasets with 'prompt' and 'completion' columns."""
    all_ds = []
    for dcfg in config.get("datasets") or []:
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

    val_cfg = config.get("validation") or {}
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


def build_teacher_prompts(dataset, config, num_proc=4):
    """Add 'teacher_prompt' column with demonstration-enriched context.

    Student prompt (CtxS): [{"role": "user", "content": <question>}]
    Teacher prompt (CtxT): [{"role": "user", "content": <question + demonstration>}]

    The 'completion' column is consumed and removed.
    """
    sdft_cfg = config.get("sdft") or {}
    teacher_template = sdft_cfg.get(
        "teacher_template",
        "{prompt}\n"
        "This is an example for a response to the question:\n"
        "{completion}\n"
        "Now answer with a response of your own, including the thinking process:",
    )
    student_system = sdft_cfg.get("student_system_prompt")
    teacher_system = sdft_cfg.get("teacher_system_prompt")

    def _format(example):
        # Student: prompt only
        student_msgs = []
        if student_system:
            student_msgs.append({"role": "system", "content": student_system})
        student_msgs.append({"role": "user", "content": example["prompt"]})

        # Teacher: prompt enriched with demonstration
        teacher_content = teacher_template.format(
            prompt=example["prompt"],
            completion=example["completion"],
        )
        teacher_msgs = []
        if teacher_system:
            teacher_msgs.append({"role": "system", "content": teacher_system})
        teacher_msgs.append({"role": "user", "content": teacher_content})

        return {"prompt": student_msgs, "teacher_prompt": teacher_msgs}

    return dataset.map(_format, num_proc=num_proc, remove_columns=["completion"])


def filter_long_prompts(dataset, tokenizer, max_prompt_length, num_proc=4):
    """Remove samples where either prompt exceeds max_prompt_length tokens."""
    initial_len = len(dataset)

    def _fits(example):
        s_text = tokenizer.apply_chat_template(
            example["prompt"], tokenize=False, add_generation_prompt=True
        )
        t_text = tokenizer.apply_chat_template(
            example["teacher_prompt"], tokenize=False, add_generation_prompt=True
        )
        s_len = len(tokenizer.encode(s_text, add_special_tokens=False))
        t_len = len(tokenizer.encode(t_text, add_special_tokens=False))
        return s_len <= max_prompt_length and t_len <= max_prompt_length

    filtered = dataset.filter(_fits, num_proc=num_proc)
    removed = initial_len - len(filtered)
    print(
        f"  Filtered out {removed}/{initial_len} samples exceeding "
        f"{max_prompt_length} prompt tokens ({len(filtered)} remaining)"
    )
    return filtered


# ── Model & Config ───────────────────────────────────────────────────────────


def load_tokenizer(config):
    model_cfg = config["model"]
    tok = AutoTokenizer.from_pretrained(
        model_cfg.get("tokenizer", model_cfg["name"]), trust_remote_code=True
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok


def load_model(config):
    model_cfg = config["model"]
    train_cfg = config.get("training") or {}
    torch_dtype = get_torch_dtype(train_cfg)

    load_kwargs = dict(trust_remote_code=True, torch_dtype=torch_dtype)
    if model_cfg.get("attn_implementation"):
        load_kwargs["attn_implementation"] = model_cfg["attn_implementation"]

    model = AutoModelForCausalLM.from_pretrained(model_cfg["name"], **load_kwargs)
    model.config.use_cache = False
    return model


def build_distil_config(config):
    train_cfg = config.get("training") or {}
    distil_cfg = config.get("distil") or {}

    use_bf16 = bool(train_cfg.get("bf16"))
    use_fp16 = bool(train_cfg.get("fp16"))
    if use_bf16 and not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()):
        use_bf16, use_fp16 = False, True

    gc_kwargs = train_cfg.get("gradient_checkpointing_kwargs")
    if train_cfg.get("gradient_checkpointing") and gc_kwargs is None:
        gc_kwargs = {"use_reentrant": False}

    kwargs = dict(
        # ── Standard training arguments ──────────────────────────────────
        output_dir=train_cfg.get("output_dir", "./outputs/sdft"),
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 1),
        per_device_eval_batch_size=train_cfg.get("per_device_eval_batch_size", 1),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 1),
        num_train_epochs=train_cfg.get("num_train_epochs", 1),
        learning_rate=float(train_cfg.get("learning_rate", 2e-5)),
        warmup_ratio=float(train_cfg.get("warmup_ratio", 0.0)),
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
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs=gc_kwargs,
        optim=train_cfg.get("optim", "adamw_torch"),
        max_grad_norm=train_cfg.get("max_grad_norm", 1.0),
        # ── Distil: generation ───────────────────────────────────────────
        max_prompt_length=distil_cfg.get("max_prompt_length", 1024),
        max_completion_length=distil_cfg.get("max_completion_length", 1024),
        num_generations=distil_cfg.get("num_generations", 1),
        temperature=distil_cfg.get("temperature", 1.0),
        top_p=distil_cfg.get("top_p", 1.0),
        top_k=distil_cfg.get("top_k"),
        min_p=distil_cfg.get("min_p"),
        repetition_penalty=distil_cfg.get("repetition_penalty", 1.0),
        generate_from_teacher=distil_cfg.get("generate_from_teacher", False),
        # ── Distil: KL / loss ────────────────────────────────────────────
        alpha=distil_cfg.get("alpha", 1.0),
        beta=distil_cfg.get("beta", 0.0),
        loss_type=distil_cfg.get("loss_type", "dapo"),
        num_loss_tokens_to_skip=distil_cfg.get("num_loss_tokens_to_skip", 0),
        mask_truncated_completions=distil_cfg.get("mask_truncated_completions", False),
        top_entropy_quantile=distil_cfg.get("top_entropy_quantile", 1.0),
        disable_dropout=distil_cfg.get("disable_dropout", True),
        # ── Distil: EMA teacher sync ─────────────────────────────────────
        sync_ref_model=distil_cfg.get("sync_ref_model", True),
        ref_model_mixup_alpha=distil_cfg.get("ref_model_mixup_alpha", 0.01),
        ref_model_sync_steps=distil_cfg.get("ref_model_sync_steps", 1),
        # ── Distil: vLLM (colocated on-policy generation) ────────────────
        use_vllm=distil_cfg.get("use_vllm", True),
        vllm_mode=distil_cfg.get("vllm_mode", "colocate"),
        vllm_gpu_memory_utilization=distil_cfg.get("vllm_gpu_memory_utilization", 0.3),
        vllm_enable_sleep_mode=distil_cfg.get("vllm_enable_sleep_mode", True),
        # ── Logging ──────────────────────────────────────────────────────
        log_completions=distil_cfg.get("log_completions", False),
    )

    if distil_cfg.get("steps_per_generation") is not None:
        kwargs["steps_per_generation"] = distil_cfg["steps_per_generation"]
    if train_cfg.get("eval_steps") is not None:
        kwargs["eval_steps"] = train_cfg["eval_steps"]

    return DistilConfig(**kwargs)


# ── Training ─────────────────────────────────────────────────────────────────


def setup_wandb(config):
    wandb_cfg = config.get("wandb") or {}
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
    print(f"[Rank {os.environ.get('RANK', 0)}] Starting SDFT training")

    setup_wandb(config)
    tokenizer = load_tokenizer(config)
    train_ds, eval_ds = load_datasets(config)

    distil_cfg = config.get("distil") or {}
    max_prompt_length = distil_cfg.get("max_prompt_length", 1024)

    # Build teacher prompts (CtxT) and format student prompts (CtxS)
    print("Formatting teacher prompts...")
    train_ds = build_teacher_prompts(train_ds, config)
    if eval_ds is not None:
        eval_ds = build_teacher_prompts(eval_ds, config)

    # # Filter long prompts (disabled for now)
    # train_ds = filter_long_prompts(train_ds, tokenizer, max_prompt_length)
    # if eval_ds is not None:
    #     eval_ds = filter_long_prompts(eval_ds, tokenizer, max_prompt_length)

    # Load student and teacher (reference) models
    print("Loading student model...")
    model = load_model(config)
    print("Loading teacher (reference) model...")
    ref_model = load_model(config)

    distil_args = build_distil_config(config)

    trainer = DistilTrainer(
        model=model,
        ref_model=ref_model,
        args=distil_args,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )

    resume_from = (config.get("training") or {}).get("resume_from_checkpoint")
    trainer.train(resume_from_checkpoint=resume_from)

    outdir = config["training"].get("output_dir", "./outputs/sdft")
    trainer.save_model(outdir)
    tokenizer.save_pretrained(outdir)
    print(f"SDFT training complete. Model saved to {outdir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
