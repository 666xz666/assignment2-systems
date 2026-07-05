import importlib.metadata

from .utils import try_gpu
from .bpe_tokenizer import BPETokenizer, BPETrainer, load_bpe_tokenizer
from .nn import (
    Linear,
    Embedding,
    RMSNorm,
    SiLU,
    SwiGLUFeedForward,
    RoPE,
    Softmax,
    ScaledDotProductAttention,
    MultiheadSelfAttention,
    TransformerLM,
    TransformerBlock,
)
from .optim import SGD, AdamW, get_lr_cosine_schedule, CosineAnnealingWarmupLR
from .utils import (
    cross_entropy,
    gradient_clipping,
    get_batch,
    create_memmap_corpus,
    load_mmap_corpus,
)
from .serialization import save_checkpoint, load_checkpoint

try:
    __version__ = importlib.metadata.version("cs336_basics")
except importlib.metadata.PackageNotFoundError:
    pass

__all__ = [
    "try_gpu",
    "BPETokenizer",
    "BPETrainer",
    "load_bpe_tokenizer",
    "Linear",
    "Embedding",
    "RMSNorm",
    "SiLU",
    "SwiGLUFeedForward",
    "RoPE",
    "Softmax",
    "ScaledDotProductAttention",
    "MultiheadSelfAttention",
    "TransformerLM",
    "TransformerBlock",
    "cross_entropy",
    "SGD",
    "AdamW",
    "get_lr_cosine_schedule",
    "CosineAnnealingWarmupLR",
    "gradient_clipping",
    "get_batch",
    "load_mmap_corpus",
    "create_memmap_corpus",
    "save_checkpoint",
    "load_checkpoint",
]
