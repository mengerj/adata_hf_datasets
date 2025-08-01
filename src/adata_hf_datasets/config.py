from dataclasses import dataclass
from typing import Optional, List


@dataclass
class Config:
    data_dir: str
    annotation_columns: str
    semantic: bool = True
    save_to_hf: bool = True


@dataclass
class TermDescriptionConfig:
    """Configuration for fetching descriptions of biomedical terms from NCBI databases."""

    config: Config
    email: Optional[str] = None
    cell_types: Optional[List[str]] = None
    diseases: Optional[List[str]] = None
    tissues: Optional[List[str]] = None
    organisms: Optional[List[str]] = None
    genes: Optional[List[str]] = None

    index: bool = True  # Whether to include index in saved CSV
    max_retries: int = 3
    batch_size: int = 20
    include_mesh_tree: bool = True  # Include MeSH tree information
    include_synonyms: bool = True  # Include alternative terms/synonyms
    search_mesh: bool = True  # Search MeSH database
    search_gene: bool = False  # Search Gene database for gene descriptions
    search_books: bool = False  # Search NCBI Bookshelf for detailed descriptions
    tokenizer_model: str = (
        "neuml/pubmedbert-base-embeddings"  # Tokenizer for token length calculation
    )
    calculate_token_length: bool = (
        True  # Whether to calculate and include token_length column
    )
    pull_descriptions: bool = False  # Must be True to actually fetch data

    def __post_init__(self):
        """Validate configuration when pull_descriptions is True."""
        if self.pull_descriptions:
            if not self.email:
                raise ValueError("Email is required when pull_descriptions=True")

            # Check if at least one term type is provided
            term_types = [
                self.cell_types,
                self.diseases,
                self.tissues,
                self.organisms,
                self.genes,
            ]
            if not any(term_types):
                raise ValueError(
                    "At least one term type (cell_types, diseases, tissues, organisms, genes) must be provided"
                )
