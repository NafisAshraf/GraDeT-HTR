"""
GraDeT-HTR training script
===========================
Multi-stage training that mirrors the paper's protocol:

  Stage 1 — sentence-level pretraining  (4.5 M synthetic sentences, 2 epochs, lr=1e-4)
  Stage 2 — word-level pretraining      (7 M   synthetic words,     1 epoch,  lr=1e-4)
  Stage 3 — real-world fine-tuning      (300 K real images,         4 epochs, lr=5e-6)

Typical usage
-------------
# Stage 1
python GraDeT_HTR/train.py --stage 1 \\
    --images_dir /data/synthetic_sentences/images \\
    --labels_file /data/synthetic_sentences/labels.csv \\
    --checkpoint_dir /checkpoints/stage1

# Stage 2 (resume from best stage-1 model)
python GraDeT_HTR/train.py --stage 2 \\
    --images_dir /data/synthetic_words/images \\
    --labels_file /data/synthetic_words/labels.csv \\
    --checkpoint_dir /checkpoints/stage2 \\
    --resume /checkpoints/stage1/best.pt

# Stage 3 (fine-tune from best stage-2 model)
python GraDeT_HTR/train.py --stage 3 \\
    --images_dir /data/real/images \\
    --labels_file /data/real/labels.csv \\
    --checkpoint_dir /checkpoints/stage3 \\
    --resume /checkpoints/stage2/best.pt
"""

import argparse
import json
import logging
import multiprocessing as mp
import os
import sys

import torch
from torch.utils.data import DataLoader

from config import DTrOCRConfig
from dataset import safe_collate, split_data
from model import DTrOCRLMHeadModel
from utils import (
    evaluate_model,
    load_checkpoint,
    save_checkpoint,
    save_final_model,
    send_inputs_to_device,
)


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging(log_dir: str) -> logging.Logger:
    """Send INFO+ logs to both stdout and a persistent file."""
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "training.log")

    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        datefmt=datefmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )
    return logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing with stage-specific defaults
# ─────────────────────────────────────────────────────────────────────────────

# Paper-specified hyperparameters per stage.
_STAGE_DEFAULTS = {
    1: dict(lr=1e-4, epochs=2,  desc="sentence pretraining"),
    2: dict(lr=1e-4, epochs=1,  desc="word pretraining"),
    3: dict(lr=5e-6, epochs=4,  desc="real-world fine-tuning"),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train GraDeT-HTR (multi-stage, paper-faithful).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Stage ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "--stage", type=int, default=1, choices=[1, 2, 3],
        help="Training stage. Sets lr/epoch defaults per the paper."
             " (1=sentence pretraining, 2=word pretraining, 3=fine-tuning)",
    )

    # ── Data ───────────────────────────────────────────────────────────────
    parser.add_argument("--images_dir",  type=str, default="../sample_train/images")
    parser.add_argument("--labels_file", type=str, default="../sample_train/labels/label.csv")
    parser.add_argument("--test_size",   type=float, default=0.05)
    parser.add_argument("--seed",        type=int,   default=42)

    # ── Training hyperparameters ────────────────────────────────────────────
    # Defaults below are overridden by --stage at the end of parse_args().
    parser.add_argument("--epochs",      type=int,   default=None,
                        help="Override the stage-default epoch count.")
    parser.add_argument("--batch_size",  type=int,   default=32,
                        help="Per-GPU batch size (paper: 32).")
    parser.add_argument("--lr",          type=float, default=None,
                        help="Override the stage-default learning rate.")
    parser.add_argument("--warmup_steps", type=int,  default=0,
                        help="Linear LR warmup steps (0 = no warmup, paper default).")
    parser.add_argument("--clip_grad_norm", type=float, default=1.0,
                        help="Max gradient norm for clipping. 0 disables clipping.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Accumulate gradients over N steps before updating.")

    # ── Early stopping ──────────────────────────────────────────────────────
    parser.add_argument("--patience", type=int, default=3,
                        help="Stop if val_loss does not improve for this many epochs. "
                             "0 disables early stopping.")

    # ── DataLoader ─────────────────────────────────────────────────────────
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--persistent_workers", action=argparse.BooleanOptionalAction, default=True)

    # ── Device / precision ─────────────────────────────────────────────────
    parser.add_argument("--device",  type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True,
                        help="Use torch.compile for faster training (Linux only).")
    parser.add_argument("--use_amp", action=argparse.BooleanOptionalAction, default=True,
                        help="Automatic mixed precision (float16).")

    # ── Checkpointing ───────────────────────────────────────────────────────
    parser.add_argument("--checkpoint_dir",    type=str, default="../checkpoints/stage1")
    parser.add_argument("--save_every",        type=int, default=1,
                        help="Save a numbered epoch checkpoint every N epochs.")
    parser.add_argument("--resume",            type=str, default="",
                        help="Path to checkpoint to resume or fine-tune from.")
    parser.add_argument("--strict_resume",     action=argparse.BooleanOptionalAction, default=True,
                        help="Strict weight loading. Use --no-strict_resume to load "
                             "partially (e.g. when architecture changed).")
    parser.add_argument("--final_model_path",  type=str, default=None,
                        help="Where to save the final model weights. "
                             "Defaults to <checkpoint_dir>/final_model.pth")

    args = parser.parse_args()

    # ── Apply stage defaults for lr / epochs if user did not override ───────
    stage_cfg = _STAGE_DEFAULTS[args.stage]
    if args.lr is None:
        args.lr = stage_cfg["lr"]
    if args.epochs is None:
        args.epochs = stage_cfg["epochs"]

    # Default checkpoint_dir per stage if user didn't set a custom one.
    if args.checkpoint_dir == "../checkpoints/stage1":
        args.checkpoint_dir = f"../checkpoints/stage{args.stage}"

    if args.final_model_path is None:
        args.final_model_path = os.path.join(args.checkpoint_dir, "final_model.pth")

    return args


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def auto_num_workers() -> int:
    count = mp.cpu_count()
    # Saturate I/O without overwhelming the scheduler.
    return min(16, max(4, count // 2))


def build_dataloader(
    dataset, batch_size, shuffle, num_workers, pin_memory,
    persistent_workers, prefetch_factor,
):
    loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=safe_collate,   # drops corrupt/missing samples gracefully
    )
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        loader_kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(dataset, **loader_kwargs)


def build_warmup_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """
    Linear warmup for the first `warmup_steps`, then constant LR.
    Returns None when warmup_steps == 0 (paper default).
    """
    if warmup_steps <= 0:
        return None

    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Logging ────────────────────────────────────────────────────────────
    logger = setup_logging(args.checkpoint_dir)
    stage_desc = _STAGE_DEFAULTS[args.stage]["desc"]
    logger.info("=" * 60)
    logger.info("GraDeT-HTR  Stage %d — %s", args.stage, stage_desc)
    logger.info("  lr=%g  epochs=%d  batch_size=%d  warmup_steps=%d",
                args.lr, args.epochs, args.batch_size, args.warmup_steps)
    logger.info("=" * 60)

    # ── Device ─────────────────────────────────────────────────────────────
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    num_workers = auto_num_workers() if args.num_workers is None else max(0, args.num_workers)
    pin_memory  = bool(args.pin_memory and device == "cuda")

    logger.info("device=%s  num_workers=%d  pin_memory=%s  use_amp=%s  compile=%s",
                device, num_workers, pin_memory, args.use_amp, args.compile)

    # ── Reproducibility ────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")

    # ── Config and datasets ────────────────────────────────────────────────
    config = DTrOCRConfig()
    logger.info("vocab_size=%d (computed from vocabulary file)", config.vocab_size)

    train_dataset, val_dataset = split_data(
        args.images_dir,
        args.labels_file,
        config,
        test_size=args.test_size,
        random_seed=args.seed,
    )
    logger.info("Train samples: %d  |  Val samples: %d",
                len(train_dataset), len(val_dataset))

    train_loader = build_dataloader(
        dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
    )
    val_loader = build_dataloader(
        dataset=val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
    )

    # ── Model ──────────────────────────────────────────────────────────────
    model = DTrOCRLMHeadModel(config)

    if args.compile:
        try:
            model = torch.compile(model)
            logger.info("torch.compile enabled.")
        except Exception as exc:
            logger.warning("torch.compile failed (%s). Continuing in eager mode.", exc)

    model.to(device)

    # ── Optimizer (Adam, paper spec) ───────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler    = torch.amp.GradScaler(device=device, enabled=args.use_amp)

    # ── LR scheduler (optional warmup only; paper uses fixed LR) ──────────
    steps_per_epoch = len(train_loader)
    total_steps     = steps_per_epoch * args.epochs
    scheduler       = build_warmup_scheduler(optimizer, args.warmup_steps, total_steps)

    # ── Checkpoint state ───────────────────────────────────────────────────
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    start_epoch   = 0
    best_val_loss = float("inf")
    epochs_no_improve = 0

    if args.resume:
        logger.info("Resuming from checkpoint: %s", args.resume)
        resume_info = load_checkpoint(
            checkpoint_path=args.resume,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            strict=args.strict_resume,
            map_location=device,
        )
        # When fine-tuning from a different stage we intentionally reset
        # epoch/history — only restore them when resuming the same stage.
        if resume_info.get("best_val_loss") is not None:
            best_val_loss = resume_info["best_val_loss"]
        if resume_info.get("history"):
            history = resume_info["history"]
            # Only resume epoch count when the run_args in the checkpoint
            # match the current stage (avoids counting stage-1 epochs as
            # stage-2 progress).
            ckpt_stage = None
            if resume_info.get("checkpoint") and resume_info["checkpoint"].get("run_args"):
                ckpt_stage = resume_info["checkpoint"]["run_args"].get("stage")
            if ckpt_stage == args.stage:
                start_epoch = int(resume_info.get("epoch", 0)) + 1
                logger.info("Same stage detected — continuing from epoch %d.", start_epoch + 1)
            else:
                logger.info(
                    "Checkpoint is from stage %s, current stage is %d — "
                    "epoch counter and history reset.",
                    ckpt_stage, args.stage,
                )
                history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
                start_epoch = 0

    # ─────────────────────────────────────────────────────────────────────
    # Training loop
    # ─────────────────────────────────────────────────────────────────────
    global_step   = start_epoch * steps_per_epoch
    config_dict   = config.__dict__.copy()
    run_args_dict = vars(args)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_losses, epoch_accuracies = [], []
        skipped_nan = 0
        skipped_oom = 0

        import tqdm as tqdm_mod
        pbar = tqdm_mod.tqdm(
            train_loader,
            total=len(train_loader),
            desc=f"Stage {args.stage} Epoch {epoch + 1}/{args.epochs}",
        )

        accum_step = 0   # counts micro-steps within one accumulation window

        for inputs in pbar:
            # safe_collate returns None when every sample in the batch is bad.
            if inputs is None:
                continue

            inputs = send_inputs_to_device(inputs, device=device)

            try:
                with torch.autocast(device_type=device, dtype=torch.float16,
                                    enabled=args.use_amp):
                    outputs = model(**inputs)

                # ── NaN / Inf loss guard ─────────────────────────────────
                if outputs.loss is None or not torch.isfinite(outputs.loss):
                    logger.warning(
                        "Non-finite loss at global_step=%d (loss=%s). Skipping batch.",
                        global_step,
                        outputs.loss.item() if outputs.loss is not None else "None",
                    )
                    skipped_nan += 1
                    optimizer.zero_grad()
                    scaler.update()   # keep scaler state consistent
                    continue

                # Scale loss for gradient accumulation so the effective
                # gradient magnitude stays constant regardless of accum count.
                scaled_loss = outputs.loss / args.gradient_accumulation_steps
                scaler.scale(scaled_loss).backward()
                accum_step += 1

                if accum_step % args.gradient_accumulation_steps == 0:
                    # ── Gradient clipping ────────────────────────────────
                    if args.clip_grad_norm > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), args.clip_grad_norm
                        )
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    if scheduler is not None:
                        scheduler.step()
                    accum_step = 0

                global_step += 1

            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    logger.warning(
                        "CUDA OOM at global_step=%d. Freeing cache and skipping batch.",
                        global_step,
                    )
                    torch.cuda.empty_cache()
                    optimizer.zero_grad()
                    scaler.update()
                    skipped_oom += 1
                    accum_step = 0
                    continue
                raise  # re-raise unexpected errors

            loss_val = outputs.loss.item()
            acc_val  = outputs.accuracy.item() if outputs.accuracy is not None else 0.0
            epoch_losses.append(loss_val)
            epoch_accuracies.append(acc_val)
            pbar.set_postfix(loss=f"{loss_val:.4f}", acc=f"{acc_val:.4f}")

        # Flush any remaining accumulated gradients at end of epoch
        if accum_step > 0:
            if args.clip_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()

        # ── Epoch-level metrics ───────────────────────────────────────────
        if not epoch_losses:
            logger.warning("Epoch %d produced no valid batches — skipping.", epoch + 1)
            continue

        train_loss = sum(epoch_losses)     / len(epoch_losses)
        train_acc  = sum(epoch_accuracies) / len(epoch_accuracies)
        val_loss, val_acc = evaluate_model(
            model, val_loader, device=device, use_amp=args.use_amp
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(
            "Epoch %d/%d  train_loss=%.4f  train_acc=%.4f  "
            "val_loss=%.4f  val_acc=%.4f  lr=%g  "
            "skipped_nan=%d  skipped_oom=%d",
            epoch + 1, args.epochs,
            train_loss, train_acc,
            val_loss, val_acc,
            current_lr,
            skipped_nan, skipped_oom,
        )

        # ── Checkpointing ─────────────────────────────────────────────────
        _ckpt_common = dict(
            model=model, optimizer=optimizer, scaler=scaler, scheduler=scheduler,
            epoch=epoch, train_loss=train_loss, val_loss=val_loss,
            train_acc=train_acc, val_acc=val_acc, best_val_loss=best_val_loss,
            history=history, config_dict=config_dict, run_args=run_args_dict,
            checkpoint_dir=args.checkpoint_dir,
        )

        # Always keep a rolling "last" checkpoint for crash recovery.
        save_checkpoint(**_ckpt_common, checkpoint_name="last.pt")

        # Numbered epoch checkpoint every N epochs.
        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(**_ckpt_common, checkpoint_name=f"epoch_{epoch + 1:04d}.pt")

        # Best-val-loss checkpoint.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            save_checkpoint(**_ckpt_common, checkpoint_name="best.pt")
            logger.info("  ↑ New best val_loss=%.4f — saved best.pt", best_val_loss)
        else:
            epochs_no_improve += 1
            logger.info(
                "  No improvement for %d epoch(s) (best=%.4f).",
                epochs_no_improve, best_val_loss,
            )

        # ── History JSON ──────────────────────────────────────────────────
        history_path = os.path.join(args.checkpoint_dir, "history.json")
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

        # ── Early stopping ────────────────────────────────────────────────
        if args.patience > 0 and epochs_no_improve >= args.patience:
            logger.info(
                "Early stopping: val_loss has not improved for %d epoch(s).",
                args.patience,
            )
            break

    # ── Final model ────────────────────────────────────────────────────────
    save_final_model(model, args.final_model_path)
    logger.info("Training complete. Final model: %s", args.final_model_path)


if __name__ == "__main__":
    main()
