import os
import yaml
import argparse
import torch
from datasets import load_dataset, concatenate_datasets, load_from_disk
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, TaskType
from trl import SFTTrainer, SFTConfig


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def is_chat_format(x):
    return (
        isinstance(x, list) and len(x) > 0 and isinstance(x[0], dict) and "role" in x[0]
    )


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


def format_dataset(ds, dcfg):
    """Normalize dataset to a format SFTTrainer understands.

    Final format is one of:
    - 'messages' column (list of {role, content}) -> chat template applied by trainer
    - 'text' column (pre-formatted string)
    """
    colmap = dcfg.get("column_map", {})
    for target, source in colmap.items():
        if (
            source != target
            and source in ds.column_names
            and target not in ds.column_names
        ):
            ds = ds.rename_column(source, target)

    if "messages" in ds.column_names:
        return ds
    if "conversations" in ds.column_names:
        return ds.rename_column("conversations", "messages")
    if "text" in ds.column_names:
        return ds

    # Convert instruction/input/output -> messages
    if "instruction" in ds.column_names:

        def to_messages(ex):
            msgs = []
            if ex.get("system"):
                msgs.append({"role": "system", "content": ex["system"]})
            user_content = ex["instruction"]
            if ex.get("input"):
                user_content += "\n" + ex["input"]
            msgs.append({"role": "user", "content": user_content})
            msgs.append(
                {
                    "role": "assistant",
                    "content": ex.get("output") or ex.get("response", ""),
                }
            )
            return {"messages": msgs}

        return ds.map(to_messages, num_proc=8, remove_columns=ds.column_names)

    raise ValueError(
        f"Cannot detect SFT format. Columns: {ds.column_names}. "
        "Expected 'messages', 'conversations', 'text', or 'instruction'."
    )


def load_datasets(config):
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

        ds = format_dataset(ds, dcfg)
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
    is_distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1
    is_fsdp = bool(train_cfg.get("fsdp"))

    if model_cfg.get("load_in_4bit") and is_fsdp:
        raise ValueError(
            "4-bit quantization (BitsAndBytes) is incompatible with FSDP. "
            "Use LoRA without quantization, or disable FSDP."
        )

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

    # Single-GPU: place directly on device.
    # DDP: no device_map, Trainer calls model.to(device) internally.
    # FSDP: no device_map, FSDP handles sharding and device placement.
    if not is_distributed and torch.cuda.is_available():
        load_kwargs["device_map"] = {"": local_rank}

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
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
        target_modules=lora_cfg.get(
            "target_modules",
            [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        ),
        use_rslora=lora_cfg.get("use_rslora", False),
    )


def build_sft_config(config):
    train_cfg = config["training"]

    use_bf16 = bool(train_cfg.get("bf16"))
    use_fp16 = bool(train_cfg.get("fp16"))
    if use_bf16 and not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()):
        print("Warning: bf16 requested but not supported, falling back to fp16")
        use_bf16, use_fp16 = False, True

    fsdp = train_cfg.get("fsdp")

    # Default to use_reentrant=False for better LoRA + grad checkpoint compat
    gc_kwargs = train_cfg.get("gradient_checkpointing_kwargs")
    if train_cfg.get("gradient_checkpointing") and gc_kwargs is None:
        gc_kwargs = {"use_reentrant": False}

    return SFTConfig(
        # Training
        output_dir=train_cfg.get("output_dir", "./outputs/sft"),
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 1),
        per_device_eval_batch_size=train_cfg.get("per_device_eval_batch_size", 1),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 1),
        num_train_epochs=train_cfg.get("num_train_epochs", 1),
        max_steps=train_cfg.get("max_steps", -1),
        learning_rate=float(train_cfg.get("learning_rate", 2e-5)),
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
        # FSDP
        fsdp=fsdp or "",
        fsdp_config=train_cfg.get("fsdp_config", {}),
        # SFT
        max_length=train_cfg.get("max_seq_length", train_cfg.get("max_length", 4096)),
        packing=train_cfg.get("packing", False),
        dataset_num_proc=train_cfg.get("dataset_num_proc"),
        dataset_text_field=train_cfg.get("dataset_text_field"),
        neftune_noise_alpha=train_cfg.get("neftune_noise_alpha"),
    )


# ── Training ─────────────────────────────────────────────────────────────────


def setup_wandb(config):
    wandb_cfg = config.get("wandb", {})
    if not wandb_cfg.get("enabled") or int(os.environ.get("RANK", 0)) != 0:
        return
    import wandb

    wandb.init(
        project=wandb_cfg.get("project", "sft-training"),
        entity=wandb_cfg.get("entity"),
        name=wandb_cfg.get("run_name"),
        tags=wandb_cfg.get("tags", []),
        config=config,
    )


def train(config):
    print(f"[Rank {os.environ.get('RANK', 0)}] Starting SFT training")

    setup_wandb(config)
    tokenizer = load_tokenizer(config)
    train_ds, eval_ds = load_datasets(config)

    model = load_model(config)
    sft_args = build_sft_config(config)
    peft_config = build_lora_config(config)

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=peft_config,
    )

    resume_from = config.get("training", {}).get("resume_from_checkpoint")
    trainer.train(resume_from_checkpoint=resume_from)

    outdir = config["training"].get("output_dir", "./outputs/sft")
    trainer.save_model(outdir)
    tokenizer.save_pretrained(outdir)
    print("SFT training complete.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
