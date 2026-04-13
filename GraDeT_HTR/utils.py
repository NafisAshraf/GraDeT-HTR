import logging
import os
import torch
from torch.utils.data import DataLoader
from typing import Tuple
import tqdm

logger = logging.getLogger(__name__)


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Return the underlying module when torch.compile has wrapped it."""
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def _atomic_torch_save(obj, path: str) -> None:
    """Write to a .tmp file then atomically rename so partial writes never
    corrupt an existing checkpoint."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp_path = f"{path}.tmp"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str,
    use_amp: bool = True,
) -> Tuple[float, float]:
    """
    Run one full pass over the validation set and return (loss, accuracy).

    - Uses torch.autocast (same dtype as training) to keep evaluation fast
      and consistent with the training forward pass.
    - Skips None batches produced by safe_collate when every sample in a
      mini-batch was corrupt.
    - Guards against an empty dataloader (returns inf loss, 0 accuracy).
    """
    model.eval()
    losses, accuracies = [], []

    with torch.no_grad():
        for inputs in tqdm.tqdm(dataloader, total=len(dataloader), desc='Evaluating'):
            if inputs is None:
                continue
            inputs = send_inputs_to_device(inputs, device=device)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                outputs = model(**inputs)

            if outputs.loss is None or not torch.isfinite(outputs.loss):
                continue
            losses.append(outputs.loss.item())
            if outputs.accuracy is not None and torch.isfinite(outputs.accuracy):
                accuracies.append(outputs.accuracy.item())

    model.train()

    if not losses:
        logger.warning("evaluate_model: no valid batches found — returning inf loss.")
        return float('inf'), 0.0

    loss = sum(losses) / len(losses)
    accuracy = sum(accuracies) / len(accuracies) if accuracies else 0.0
    return loss, accuracy


def send_inputs_to_device(dictionary, device):
    return {
        key: value.to(device=device) if isinstance(value, torch.Tensor) else value
        for key, value in dictionary.items()
    }


def save_checkpoint(
    model,
    optimizer,
    epoch,
    train_loss,
    val_loss,
    train_acc,
    val_acc,
    checkpoint_dir,
    checkpoint_name,
    scaler=None,
    scheduler=None,
    best_val_loss=None,
    history=None,
    config_dict=None,
    run_args=None,
):
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': _unwrap_model(model).state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_acc': train_acc,
        'val_acc': val_acc,
        'best_val_loss': best_val_loss,
        'history': history,
        'config': config_dict,
        'run_args': run_args,
    }
    if scaler is not None:
        checkpoint['scaler_state_dict'] = scaler.state_dict()
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    _atomic_torch_save(checkpoint, checkpoint_path)
    logger.info("Checkpoint saved to %s", checkpoint_path)
    return checkpoint_path


def load_checkpoint(
    checkpoint_path,
    model,
    optimizer=None,
    scaler=None,
    scheduler=None,
    strict=True,
    map_location='cpu',
):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # weights_only=False is required because checkpoints contain optimizer
    # state dicts which use Python objects beyond plain tensors.
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model_state_dict = checkpoint['model_state_dict']
    else:
        model_state_dict = checkpoint

    _unwrap_model(model).load_state_dict(model_state_dict, strict=strict)

    if optimizer is not None and isinstance(checkpoint, dict) and checkpoint.get('optimizer_state_dict'):
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scaler is not None and isinstance(checkpoint, dict) and checkpoint.get('scaler_state_dict'):
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    if scheduler is not None and isinstance(checkpoint, dict) and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    epoch        = checkpoint.get('epoch', 0)       if isinstance(checkpoint, dict) else 0
    train_loss   = checkpoint.get('train_loss')     if isinstance(checkpoint, dict) else None
    val_loss     = checkpoint.get('val_loss')       if isinstance(checkpoint, dict) else None
    train_acc    = checkpoint.get('train_acc')      if isinstance(checkpoint, dict) else None
    val_acc      = checkpoint.get('val_acc')        if isinstance(checkpoint, dict) else None
    best_val_loss = checkpoint.get('best_val_loss') if isinstance(checkpoint, dict) else None
    history      = checkpoint.get('history')        if isinstance(checkpoint, dict) else None

    logger.info("Resumed from epoch %d", epoch)
    if train_loss is not None and val_loss is not None:
        logger.info("  train_loss=%.4f  val_loss=%.4f", train_loss, val_loss)
    if train_acc is not None and val_acc is not None:
        logger.info("  train_acc=%.4f   val_acc=%.4f", train_acc, val_acc)

    return {
        'epoch': epoch,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_acc': train_acc,
        'val_acc': val_acc,
        'best_val_loss': best_val_loss,
        'history': history,
        'checkpoint': checkpoint if isinstance(checkpoint, dict) else None,
    }


def save_final_model(model, final_model_path):
    unwrapped = _unwrap_model(model)
    state_dict_cpu = {k: v.detach().cpu() for k, v in unwrapped.state_dict().items()}
    _atomic_torch_save(state_dict_cpu, final_model_path)
    logger.info("Final model saved to %s", final_model_path)


def load_final_model(model, final_model_path):
    if not os.path.isfile(final_model_path):
        raise FileNotFoundError(f"Final model not found: {final_model_path}")
    state_dict = torch.load(final_model_path, map_location='cpu', weights_only=False)
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    # Strip the _orig_mod. prefix added by torch.compile if present.
    state_dict = {
        k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v
        for k, v in state_dict.items()
    }
    model.load_state_dict(state_dict)
    return model
