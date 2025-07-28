#!/usr/bin/env python3
"""
Script to fetch descriptions for biomedical terms from NCBI databases.

This script demonstrates how to use the gen_term_descriptions function to:
1. Fetch descriptions from NCBI MeSH database for cell types, diseases, tissues, etc.
2. Save results in separate CSV files organized by term type
3. Include synonyms, MeSH tree information, and source database info
4. Create HuggingFace datasets with anchor-positive pairs for sentence transformer training
5. Upload datasets to HuggingFace Hub for easy access and sharing

Requirements:
- Valid email address for NCBI API access
- Terms to search for (cell types, diseases, tissues, organisms, genes)
- HuggingFace account and valid token (in hf_config.py) for dataset upload
"""

from adata_hf_datasets.config import Config, TermDescriptionConfig
from adata_hf_datasets.term_descriptions import gen_term_descriptions


def main():
    """
    Example usage of gen_term_descriptions function.

    Modify the parameters below according to your needs:
    """

    # ========== CONFIGURATION PARAMETERS ==========

    # Basic configuration
    data_dir = "term_descriptions_example"  # Output folder name in out/data/

    # NCBI API configuration
    email = "jonatan.menger@uniklinik-freiburg.de"  # REQUIRED: Your email for NCBI API access

    # Define your terms of interest
    cell_types = [
        "T cell",
        "B cell",
        "NK cell",
        "macrophage",
        "neutrophil",
        "dendritic cell",
        "monocyte",
        "eosinophil",
    ]
    """
    diseases = [
        "cancer",
        "diabetes mellitus",
        "Alzheimer disease",
        "rheumatoid arthritis",
        "Crohn disease",
        "multiple sclerosis"
    ]

    tissues = [
        "liver",
        "kidney",
        "brain",
        "heart",
        "lung",
        "spleen"
    ]

    organisms = [
        "Homo sapiens"  # Human
    ]

    genes = [
        "CD4",
        "CD8A",
        "IL2",
        "TNF",
        "FOXP3"
    ]
    """
    # Load cell types from file
    cell_types_path = "resources/cell_types.txt"
    with open(cell_types_path, "r") as file:
        cell_types = file.read().splitlines()
    gene_path = "resources/genes.txt"
    with open(gene_path, "r") as file:
        genes = file.read().splitlines()
    # ========== MANUAL TERM LIST OPTIONS ==========
    diseases = None
    tissues = None
    organisms = None
    cell_types = None

    # Advanced settings (OPTIMIZED FOR SPEED)
    max_retries = 1  # Reduced retries for speed
    batch_size = 50  # Larger batches for efficiency
    include_mesh_tree = False  # Skip for speed
    include_synonyms = False  # Skip for speed (major time saver)

    # Database options (GENES ONLY - skip slow searches)
    search_mesh = False  # Skip MeSH for genes (use Gene DB directly)
    search_gene = True  # Only use Gene database for genes
    search_books = False  # Skip slow Books database

    # Token length calculation options (DISABLED FOR SPEED)
    tokenizer_model = "neuml/pubmedbert-base-embeddings"  # Not used when disabled
    calculate_token_length = False  # ❌ DISABLED - Major speedup!

    # ========== CREATE CONFIGURATION OBJECTS ==========

    # Create base configuration
    config = Config(
        data_dir=data_dir,
        annotation_columns=["term_type"],  # This is just for compatibility
        semantic=False,
        save_to_hf=True,  # ✅ Set to True to upload anchor-positive pairs to HuggingFace
    )

    # Create term description configuration
    description_config = TermDescriptionConfig(
        config=config,
        email=email,
        cell_types=cell_types,
        diseases=diseases,
        tissues=tissues,
        organisms=organisms,
        genes=genes,
        # Auto-fetch options
        max_retries=max_retries,
        batch_size=batch_size,
        include_mesh_tree=include_mesh_tree,
        include_synonyms=include_synonyms,
        search_mesh=search_mesh,
        search_gene=search_gene,
        search_books=search_books,
        tokenizer_model=tokenizer_model,
        calculate_token_length=calculate_token_length,
        pull_descriptions=True,  # This MUST be True to actually fetch data
    )

    # ========== VALIDATE CONFIGURATION ==========

    print("Configuration Summary:")
    print(f"  Data directory: {data_dir}")
    print(f"  Email: {email}")
    print()
    print("📝 Manual Terms:")
    print(f"  Cell types: {len(cell_types) if cell_types else 0} terms")
    print(f"  Diseases: {len(diseases) if diseases else 0} terms")
    print(f"  Tissues: {len(tissues) if tissues else 0} terms")
    print(f"  Organisms: {len(organisms) if organisms else 0} terms")
    print(f"  Genes: {len(genes) if genes else 0} terms")
    print()

    print("⚙️ Advanced Settings:")
    print(f"  Search MeSH: {search_mesh}")
    print(f"  Search Gene DB: {search_gene}")
    print(f"  Search Books: {search_books}")
    print(f"  Include synonyms: {include_synonyms}")
    print(f"  Include MeSH tree: {include_mesh_tree}")
    print(f"  Calculate token lengths: {calculate_token_length}")
    if calculate_token_length:
        print(f"  Tokenizer model: {tokenizer_model}")
    print()

    # Basic validation
    if email == "your.email@university.edu":
        print("❌ ERROR: Please update the email address in the script!")
        return

    # ========== RUN THE FUNCTION ==========

    print("🚀 Starting term description fetching...")
    print("This may take several minutes depending on the number of terms.")
    print("The function will search NCBI databases for detailed descriptions.")
    print()

    try:
        # This is the main function call
        gen_term_descriptions(description_config)

        print()
        print("✅ Success! Term descriptions have been fetched and saved.")
        print(f"📁 Check the output in: out/data/{data_dir}/descriptions/")
        print("   Separate CSV files for each term type:")
        print("   - cell_types_descriptions.csv")
        print("   - diseases_descriptions.csv")
        print("   - tissues_descriptions.csv")
        print("   - organisms_descriptions.csv")
        print("   - genes_descriptions.csv")
        print()
        print("Each CSV contains columns: term, term_type, description, mesh_id,")
        print("synonyms, mesh_tree, source_database, definition_source, token_length")
        print()
        print(
            "🚀 HuggingFace datasets have also been created with anchor-positive pairs!"
        )
        print("   Check your HuggingFace account for datasets named:")
        print(f"   - {data_dir}_term_descriptions_cell_types_anchor_positive")
        print(f"   - {data_dir}_term_descriptions_diseases_anchor_positive")
        print(f"   - {data_dir}_term_descriptions_tissues_anchor_positive")
        print(f"   - {data_dir}_term_descriptions_organisms_anchor_positive")
        print(f"   - {data_dir}_term_descriptions_genes_anchor_positive")
        print()
        print("📊 Each HuggingFace dataset contains:")
        print("   • anchor: term (including synonyms)")
        print("   • positive: clean ontology description")
        print("   • source_database: database name with proper attribution")
        print("   • source_link: direct link to the ontology")
        print("   • is_synonym: boolean flag for synonym pairs")
        print("   • comprehensive README with examples and citations")

    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")
        print("Please check your configuration and try again.")
        raise


def simple_example():
    """
    Minimal example with just a few terms.
    """
    # Basic setup
    config = Config(data_dir="simple_descriptions", annotation_columns=["term_type"])

    description_config = TermDescriptionConfig(
        config=config,
        email="your.email@university.edu",  # UPDATE THIS
        cell_types=["T cell", "macrophage"],
        diseases=["cancer", "diabetes"],
        pull_descriptions=True,
    )

    gen_term_descriptions(description_config)


if __name__ == "__main__":
    # Run the main example
    main()

    # Uncomment the line below to run the simple example instead
    # simple_example()
