from pathlib import Path
from typing import Optional, Union, Tuple, List, Literal

# Absolute path to the repo root, resolved relative to this file so the code
# works regardless of the working directory when train.py is launched.
_REPO_ROOT = Path(__file__).parent.parent
_DEFAULT_VOCAB_FILE = str(_REPO_ROOT / 'tokenization' / 'bn_grapheme_1296_from_bengali.ai.buet.txt')


class DTrOCRConfig:
    def __init__(
        self,
        gpt2_hf_model: str = 'openai-community/gpt2',
        vit_hf_model: str = 'google/vit-base-patch16-224',
        # vocab_size is computed dynamically from the vocab file when None,
        # ensuring the embedding table always matches the actual vocabulary.
        vocab_size: Optional[int] = None,
        max_position_embeddings: Optional[int] = 256,
        hidden_size: Optional[int] = 768,
        num_hidden_layers: Optional[int] = 12,
        num_attention_heads: Optional[int] = 12,
        patch_size: Optional[Union[Tuple[int, int], List[int]]] = (4, 8),    # (height, width)
        image_size: Optional[Union[Tuple[int, int], List[int]]] = (32, 128), # (height, width)
        num_channels: Optional[int] = 3,
        resid_pdrop: Optional[float] = 0.1,
        embd_pdrop: Optional[float] = 0.1,
        attn_pdrop: Optional[float] = 0.1,
        layer_norm_epsilon: Optional[float] = 1e-5,
        attn_implementation: Literal['sdpa', 'flash_attention_2'] = 'sdpa',
        bn_vocab_file: str = _DEFAULT_VOCAB_FILE,
    ):
        self.gpt2_hf_model = gpt2_hf_model
        self.vit_hf_model = vit_hf_model
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.patch_size = patch_size
        self.image_size = image_size
        self.num_channels = num_channels
        self.max_position_embeddings = max_position_embeddings
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self._attn_implementation = attn_implementation
        self.bn_vocab_file = bn_vocab_file

        if vocab_size is None:
            self.vocab_size = self._compute_vocab_size()
        else:
            self.vocab_size = vocab_size

        # GPT-2 config fields expected by transformers internals
        self.n_inner = None
        self.scale_attn_weights = True
        self.scale_attn_by_inverse_layer_idx = False
        self.reorder_and_upcast_attn = False
        self.add_cross_attention = False
        self.activation_function = "gelu_new"

    def _compute_vocab_size(self) -> int:
        """
        Count actual vocabulary size from the grapheme file so the embedding
        table is always exactly the right size.

        GraphemeTokenizer initialises with vocab = ["▁" (oov, id=0), "_" (pad, id=1)].
        add_tokens() then appends every grapheme from the file that is not already
        present.  We mirror that logic here.
        """
        vocab_path = Path(self.bn_vocab_file)
        if not vocab_path.exists():
            raise FileNotFoundError(
                f"Vocabulary file not found: {self.bn_vocab_file}\n"
                f"Expected at: {vocab_path.resolve()}\n"
                f"Make sure the tokenization/ directory is present at the repo root."
            )
        with open(vocab_path, 'r', encoding='utf-8') as f:
            graphemes = sorted(list(set(
                line.strip() for line in f if line.strip()
            )))
        # Mirror _validate_tokens: exclude tokens already in the initial vocab.
        initial = {"▁", "_"}
        unique_graphemes = [g for g in graphemes if g not in initial]
        # 2 initial tokens + unique graphemes from file
        return 2 + len(unique_graphemes)
