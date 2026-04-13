import logging
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from config import DTrOCRConfig
from processor import DTrOCRProcessor

logger = logging.getLogger(__name__)


class HandwrittenDataset(Dataset):
    def __init__(self, images_dir, data_frame, config: DTrOCRConfig, indices=None):
        super().__init__()
        self.images_dir = images_dir
        # Store a reset-indexed copy so iloc lookups are always valid.
        self.df = data_frame.reset_index(drop=True)
        self.indices = np.arange(len(self.df)) if indices is None else np.asarray(indices)
        self.processor = DTrOCRProcessor(config, add_eos_token=True, add_bos_token=True)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        row = self.df.iloc[self.indices[index]]

        # ── text validation ────────────────────────────────────────────────────
        raw_text = row["text"]
        if pd.isna(raw_text):
            logger.warning("Skipping sample %d: text is NaN.", index)
            return None
        text = str(raw_text).strip()
        if not text or text.lower() == "nan":
            logger.warning("Skipping sample %d: text is empty or 'nan'.", index)
            return None

        # ── image loading ──────────────────────────────────────────────────────
        image_path = os.path.join(self.images_dir, str(row["image_id"]))
        try:
            image = Image.open(image_path).convert('RGB')
        except FileNotFoundError:
            logger.warning("Skipping sample %d: image not found at '%s'.", index, image_path)
            return None
        except Exception as exc:
            logger.warning("Skipping sample %d: failed to load image '%s': %s", index, image_path, exc)
            return None

        # ── processor ─────────────────────────────────────────────────────────
        try:
            inputs = self.processor(
                images=image,
                texts=text,
                padding=True,
                return_tensors="pt",
                return_labels=True,
            )
        except Exception as exc:
            logger.warning(
                "Skipping sample %d (text='%s...'): processor error: %s",
                index, text[:30], exc,
            )
            return None

        return {
            'pixel_values':   inputs.pixel_values[0],
            'input_ids':      inputs.input_ids[0],
            'attention_mask': inputs.attention_mask[0],
            'labels':         inputs.labels[0],
        }


def safe_collate(batch):
    """
    Collate function that silently drops None entries returned by __getitem__
    for corrupt/missing images or invalid text.  Returns None if every sample
    in the batch was bad so the training loop can skip the step cleanly.
    """
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


def split_data(images_dir, labels_file, config, test_size=0.05, random_seed=42):
    """
    Load the CSV and return (train_dataset, val_dataset).

    Pre-validates paths and required columns before touching the model so
    errors are caught early with a helpful message.
    """
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(
            f"Images directory not found: {images_dir}\n"
            f"Pass --images_dir pointing to your image folder."
        )
    if not os.path.isfile(labels_file):
        raise FileNotFoundError(
            f"Labels CSV not found: {labels_file}\n"
            f"Pass --labels_file pointing to your CSV with 'image_id' and 'text' columns."
        )

    df = pd.read_csv(labels_file, encoding='utf-8')

    required_cols = {'image_id', 'text'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Labels CSV is missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    indices = np.arange(len(df))
    train_indices, test_indices = train_test_split(
        indices, test_size=test_size, random_state=random_seed
    )

    # Share one DataFrame between both splits to avoid doubling RAM usage.
    train_dataset = HandwrittenDataset(images_dir, df, config, train_indices)
    test_dataset  = HandwrittenDataset(images_dir, df, config, test_indices)

    return train_dataset, test_dataset
