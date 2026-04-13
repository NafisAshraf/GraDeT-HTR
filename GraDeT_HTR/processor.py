import logging
import os
from pathlib import Path
from typing import List, Union

from PIL import Image
from transformers import AutoImageProcessor

from bntokenizer import BnGraphemizerProcessor
from config import DTrOCRConfig
from data import DTrOCRProcessorOutput

logger = logging.getLogger(__name__)


class DTrOCRProcessor:
    def __init__(
        self,
        config: DTrOCRConfig,
        add_bos_token: bool = False,
        add_eos_token: bool = False,
    ):
        # ── ViT image processor ───────────────────────────────────────────────
        # Try local cache first so the code works on GPU servers that may have
        # limited or no outbound internet access.
        vit_kwargs = dict(
            size={"height": config.image_size[0], "width": config.image_size[1]},
            use_fast=True,
        )
        try:
            self.vit_processor = AutoImageProcessor.from_pretrained(
                config.vit_hf_model, **vit_kwargs
            )
        except OSError:
            # Retry in offline mode (uses the local HuggingFace cache).
            logger.warning(
                "Could not reach HuggingFace Hub. Retrying with local_files_only=True."
            )
            self.vit_processor = AutoImageProcessor.from_pretrained(
                config.vit_hf_model, local_files_only=True, **vit_kwargs
            )

        # ── Bengali grapheme tokenizer ────────────────────────────────────────
        # model_max_length = max sequence positions minus image patch tokens.
        num_patches = int(
            (config.image_size[0] / config.patch_size[0])
            * (config.image_size[1] / config.patch_size[1])
        )
        text_max_length = config.max_position_embeddings - num_patches
        if text_max_length <= 0:
            raise ValueError(
                f"image_size={config.image_size} / patch_size={config.patch_size} yields "
                f"{num_patches} patches which is >= max_position_embeddings="
                f"{config.max_position_embeddings}.  Reduce patch count or increase "
                f"max_position_embeddings."
            )

        self.tokeniser = BnGraphemizerProcessor(
            config.bn_vocab_file,
            add_bos_token=add_bos_token,
            add_eos_token=add_eos_token,
            model_max_length=text_max_length,
        )

    def __call__(
        self,
        images: Union[Image.Image, List[Image.Image]] = None,
        texts: Union[str, List[str]] = None,
        return_labels: bool = False,
        input_data_format: str = 'channels_last',
        padding: Union[bool, str] = False,
        *args,
        **kwargs,
    ) -> DTrOCRProcessorOutput:
        text_inputs = (
            self.tokeniser(texts, padding=padding)
            if texts is not None else None
        )
        image_inputs = (
            self.vit_processor(images, input_data_format=input_data_format, *args, **kwargs)
            if images is not None else None
        )

        return DTrOCRProcessorOutput(
            pixel_values=image_inputs["pixel_values"] if image_inputs is not None else None,
            input_ids=text_inputs['input_ids'] if text_inputs is not None else None,
            attention_mask=text_inputs['attention_mask'] if text_inputs is not None else None,
            labels=text_inputs['input_ids'] if (text_inputs is not None and return_labels) else None,
        )
