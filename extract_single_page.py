import logging
import os
import time

import torch
from PIL import Image
from tqdm import tqdm

from GraDeT_HTR.config import DTrOCRConfig
from GraDeT_HTR.model import DTrOCRLMHeadModel
from GraDeT_HTR.processor import DTrOCRProcessor
from GraDeT_HTR.utils import load_final_model

logger = logging.getLogger(__name__)

ENG2BEN_MAP = str.maketrans({
    "0": "০", "1": "১", "2": "২", "3": "৩", "4": "৪",
    "5": "৫", "6": "৬", "7": "৭", "8": "৮", "9": "৯",
})


def load_extraction_model(
    root_path='',
    weights="checkpoints/final_model.pth",
    device="cpu",
):
    # DTrOCRConfig resolves the vocab file path absolutely via pathlib, so
    # we no longer need to pass bn_vocab_file explicitly.
    config = DTrOCRConfig()
    model  = DTrOCRLMHeadModel(config)
    model  = load_final_model(model, weights)
    model.eval()
    model.to(device)

    processor = DTrOCRProcessor(config)

    if device != "cpu":
        logger.info("Extraction model on: %s", next(model.parameters()).device)

    return model, processor


# ── sort helper ────────────────────────────────────────────────────────────────

def sort_underscore_numbers(keys):
    """Sort strings like '1_1_1_10', '1_1_1_2' by their numeric components."""
    return sorted(keys, key=lambda s: tuple(int(p) for p in s.split('_')))


# ── single-word inference ──────────────────────────────────────────────────────

def extract_word_text(path_to_image, model, processor, device="cpu"):
    """
    Run model+processor on one word image and return the decoded text.
    Returns an empty string on any image-loading or inference error so a
    single bad image never interrupts a whole-page extraction run.
    """
    try:
        image = Image.open(path_to_image).convert("RGB")
    except FileNotFoundError:
        logger.warning("Image not found: %s", path_to_image)
        return ""
    except Exception as exc:
        logger.warning("Failed to load image '%s': %s", path_to_image, exc)
        return ""

    try:
        inputs = processor(
            images=image,
            texts=processor.tokeniser.bos_token,
            return_tensors="pt",
        )

        if inputs.pixel_values is not None:
            inputs.pixel_values = inputs.pixel_values.to(device)
        if inputs.input_ids is not None:
            inputs.input_ids = inputs.input_ids.to(device)
        if inputs.attention_mask is not None:
            inputs.attention_mask = inputs.attention_mask.to(device)

        model_output = model.generate(
            inputs=inputs,
            processor=processor,
            num_beams=3,
            use_cache=True,
        )
        decoded = processor.tokeniser.decode(model_output[0])
        # Strip OOV/BOS/EOS token "▁" and leading/trailing whitespace,
        # then remap ASCII digits to Bengali digits.
        return decoded.replace("▁", "").strip().translate(ENG2BEN_MAP)

    except Exception as exc:
        logger.warning("Inference failed for '%s': %s", path_to_image, exc)
        return ""


# ── process one line directory ─────────────────────────────────────────────────

def process_line_dir(line_dir, model, processor, device="cpu"):
    """
    Process all word images in a line directory (e.g. 1_1_1/).
    """
    files = [
        f for f in os.listdir(line_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
    base2file = {os.path.splitext(f)[0]: f for f in files}
    sorted_bases = sort_underscore_numbers(base2file.keys())

    words = []
    for base in sorted_bases:
        img_path = os.path.join(line_dir, base2file[base])
        words.append(extract_word_text(img_path, model, processor, device=device))

    return " ".join(words)


# ── process entire page ────────────────────────────────────────────────────────

def process_page_dir(page_dir, model, processor, device="cpu"):
    """
    Process all line subdirectories inside a page directory (e.g. '1_1').
    """
    subdirs = [
        d for d in os.listdir(page_dir)
        if os.path.isdir(os.path.join(page_dir, d))
    ]
    sorted_lines = sort_underscore_numbers(subdirs)

    lines = []
    for line in sorted_lines:
        line_path = os.path.join(page_dir, line)
        lines.append(process_line_dir(line_path, model, processor, device))

    return "\n".join(lines)


def extract_full_page(page_dir, model, processor, device="cpu"):
    return process_page_dir(page_dir, model, processor, device=device)


# ── standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, processor = load_extraction_model(device=device)

    PAGE_DIR = "BN_DRISHTI/content/final_word_segmentation"

    start = time.time()
    torch.cuda.reset_peak_memory_stats()

    full_text = extract_full_page(PAGE_DIR, model, processor, device=device)
    print(full_text)

    elapsed = time.time() - start
    print(f"Total time: {elapsed:.2f}s")
    peak_gb = torch.cuda.max_memory_allocated() / 1024 ** 3
    print(f"Peak GPU memory: {peak_gb:.2f} GB")
