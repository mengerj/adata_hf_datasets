# adata_hf_datasets/term_fetches/__init__.py

from .term_descriptions import gen_term_descriptions
from .config import Config, TermDescriptionConfig

__all__ = [
    "gen_term_descriptions",
    "Config",
    "TermDescriptionConfig",
]
