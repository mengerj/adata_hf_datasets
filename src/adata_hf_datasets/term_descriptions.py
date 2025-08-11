import time
import re
import requests
import pandas as pd
from tqdm import tqdm
from Bio import Entrez
from pathlib import Path
from datasets import Dataset
from adata_hf_datasets.hf_config import hf_config
import logging
import mygene


logger = logging.getLogger(__name__)


def save_to_csv(dataframe, dataframe_name, config, subfolder):
    current_dir = Path(__file__).resolve().parent.parent
    output_dir = current_dir / "out" / "data" / config.config.data_dir / subfolder
    output_dir.mkdir(parents=True, exist_ok=True)

    dataframe.to_csv(output_dir / dataframe_name, index=config.index)

    print(f"Dataframe saved to: {output_dir / dataframe_name}")


def fetch_term_descriptions(terms, term_type, description_config):
    """
    Fetch descriptions for biomedical terms from NCBI databases.

    Args:
        terms (List[str]): List of terms to search for
        term_type (str): Type of terms ('cell_types', 'diseases', 'tissues', 'organisms', 'genes')
        description_config: TermDescriptionConfig object

    Returns:
        pd.DataFrame: DataFrame with term descriptions and metadata
    """
    Entrez.email = description_config.email
    results = []

    # Load tokenizer if token length calculation is enabled
    tokenizer = None
    if description_config.calculate_token_length:
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                description_config.tokenizer_model
            )
            print(f"Loaded tokenizer: {description_config.tokenizer_model}")
        except Exception as e:
            print(
                f"Warning: Could not load tokenizer {description_config.tokenizer_model}: {e}"
            )
            print("Token lengths will not be calculated.")

    print(f"Fetching descriptions for {len(terms)} {term_type}...")

    # Process terms in batches
    for i in tqdm(range(0, len(terms), description_config.batch_size)):
        batch_terms = terms[i : i + description_config.batch_size]

        for term in batch_terms:
            term_data = {
                "term": term,
                "term_type": term_type,
                "description": "",
                "mesh_id": "",
                "synonyms": "",
                "mesh_tree": "",
                "source_database": "",
                "definition_source": "",
            }

            # Try ontologies first, then MeSH database
            if description_config.search_mesh:
                # First try ontologies with proper term type
                ontology_data = fetch_from_ontologies(
                    term, term_type, description_config
                )
                if ontology_data:
                    term_data.update(ontology_data)
                    term_data["source_database"] = ontology_data["definition_source"]
                else:
                    # Fallback to the old approach
                    mesh_data = fetch_from_mesh(term, description_config)
                    if mesh_data:
                        term_data.update(mesh_data)
                        term_data["source_database"] = mesh_data["definition_source"]

            # Try Gene database for genes
            if description_config.search_gene and term_type == "genes":
                gene_data = fetch_from_gene_db(term, description_config)
                if gene_data and not term_data["description"]:
                    term_data.update(gene_data)
                    term_data["source_database"] = "Gene"

            # Try Books database for comprehensive descriptions
            if description_config.search_books and not term_data["description"]:
                books_data = fetch_from_books(term, description_config)
                if books_data:
                    term_data.update(books_data)
                    term_data["source_database"] = "Books"

            # Calculate token length if tokenizer is available and description exists
            if (
                tokenizer
                and term_data["description"]
                and description_config.calculate_token_length
            ):
                try:
                    tokens = tokenizer.encode(
                        term_data["description"], add_special_tokens=False
                    )
                    term_data["token_length"] = len(tokens)
                except Exception as e:
                    print(
                        f"Warning: Could not calculate token length for '{term}': {e}"
                    )
                    term_data["token_length"] = 0
            else:
                term_data["token_length"] = (
                    0 if description_config.calculate_token_length else None
                )

            results.append(term_data)

            # Add minimal delay only for API requests (much faster!)
            # For large batches, use minimal delay to avoid overwhelming servers
            if len(results) % 50 == 0:  # Only delay every 50 terms
                time.sleep(0.01)  # Reduced from 0.1-0.3 to 0.01 seconds!

    # Create DataFrame and add summary statistics if token lengths were calculated
    df = pd.DataFrame(results)

    if description_config.calculate_token_length and tokenizer:
        max_token_length = df["token_length"].max()
        avg_token_length = df["token_length"].mean()
        print("Token length statistics:")
        print(f"  Maximum: {max_token_length}")
        print(f"  Average: {avg_token_length:.1f}")
        print(f"  Tokenizer: {description_config.tokenizer_model}")

    return df


def fetch_from_ontologies(term, term_type, description_config):
    """Fetch term information from biomedical ontologies via EBI OLS API."""

    # Define which ontologies to search based on term type
    ontology_mapping = {
        "cell_types": [
            "cl"
        ],  # Cell Ontology ONLY (remove GO to avoid molecular functions)
        "diseases": ["mondo", "hp", "doid"],  # MONDO, Human Phenotype, Disease Ontology
        "tissues": ["uberon", "fma"],  # Uberon, Foundational Model of Anatomy
        "organisms": ["ncbitaxon"],  # NCBI Taxonomy
        "genes": ["go", "pr"],  # Gene Ontology, Protein Ontology
    }

    ontologies = ontology_mapping.get(term_type, ["cl", "mondo", "uberon"])

    for ontology in ontologies:
        try:
            # Search in the specific ontology
            search_url = "https://www.ebi.ac.uk/ols/api/search"
            params = {
                "q": term,
                "ontology": ontology,
                "rows": 5,
                "exact": "false",
                "groupField": "iri",
                "start": 0,
            }

            response = requests.get(search_url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()

                if data["response"]["docs"]:
                    # Get the best match
                    best_match = data["response"]["docs"][0]

                    # Extract clean information
                    description = (
                        best_match.get("description", [""])[0]
                        if best_match.get("description")
                        else ""
                    )

                    # If no description, try obo_definition_citation
                    if not description:
                        description = (
                            best_match.get("obo_definition_citation", [""])[0]
                            if best_match.get("obo_definition_citation")
                            else ""
                        )

                    # Get synonyms
                    synonyms = []
                    if description_config.include_synonyms:
                        synonym_sources = ["obo_synonym", "synonym", "alternative_term"]
                        for source in synonym_sources:
                            if best_match.get(source):
                                synonyms.extend(
                                    best_match[source][:3]
                                )  # Limit to 3 synonyms

                    # Clean up description (remove citation brackets, extra whitespace)
                    if description:
                        description = re.sub(
                            r"\[.*?\]", "", description
                        )  # Remove citation brackets
                        description = re.sub(
                            r"\s+", " ", description
                        ).strip()  # Clean whitespace

                        result = {
                            "mesh_id": best_match.get(
                                "obo_id", best_match.get("iri", "N/A")
                            ),
                            "description": description,
                            "synonyms": "; ".join(synonyms) if synonyms else "",
                            "mesh_tree": ontology.upper(),
                            "definition_source": f"{ontology.upper()} Ontology",
                        }

                        return result

            # Minimal delay between ontology requests (optimized)
            time.sleep(0.005)  # Reduced from 0.2 to 0.005 seconds

        except Exception as e:
            print(f"Error searching {ontology} for '{term}': {e}")
            continue

    return None


def fetch_from_mesh(term, description_config):
    """Fetch term information using ontologies first, then NCBI as fallback."""

    # First try ontologies (much cleaner definitions)
    ontology_result = fetch_from_ontologies(
        term, "cell_types", description_config
    )  # Default to cell_types
    if ontology_result:
        return ontology_result

    # Fallback to NCBI - but get cleaner data
    try:
        Entrez.email = description_config.email

        # Try MedlinePlus or medical dictionary entries first
        search_query = f'"{term}"[Title] AND (dictionary OR definition OR glossary)'

        search_handle = Entrez.esearch(db="pubmed", term=search_query, retmax=2)
        search_results = Entrez.read(search_handle)
        search_handle.close()

        if search_results["IdList"]:
            pmid = search_results["IdList"][0]

            # Get the title and abstract
            fetch_handle = Entrez.efetch(
                db="pubmed", id=pmid, rettype="abstract", retmode="text"
            )
            full_text = fetch_handle.read()
            fetch_handle.close()

            if full_text:
                # Try to extract just the definition part
                lines = full_text.split("\n")
                clean_lines = []

                for line in lines:
                    line = line.strip()
                    # Skip author info, journal info, etc.
                    if any(
                        skip_word in line.lower()
                        for skip_word in [
                            "author",
                            "pmid:",
                            "doi:",
                            "©",
                            "copyright",
                            "affiliat",
                        ]
                    ):
                        break
                    # Keep lines that look like definitions
                    if len(line) > 20 and not line.endswith("."):  # Not just a title
                        clean_lines.append(line)

                if clean_lines:
                    description = " ".join(clean_lines[:2])  # First 2 meaningful lines
                    description = (
                        description[:300] + "..."
                        if len(description) > 300
                        else description
                    )

                    return {
                        "mesh_id": f"PMID_{pmid}",
                        "description": description,
                        "synonyms": "",
                        "mesh_tree": "",
                        "definition_source": "NCBI Dictionary",
                    }

        # Final fallback - create a basic description
        return {
            "mesh_id": "N/A",
            "description": f"{term}: A biomedical term. Detailed definition not available from standard databases.",
            "synonyms": "",
            "mesh_tree": "",
            "definition_source": "Basic",
        }

    except Exception as e:
        print(f"Error fetching data for '{term}': {e}")
        return {
            "mesh_id": "ERROR",
            "description": f"{term}: Error occurred while fetching description.",
            "synonyms": "",
            "mesh_tree": "",
            "definition_source": "Error",
        }


def fetch_from_gene_db(term, description_config):
    """Fetch gene information using mygene for gene ID + NCBI API for description."""
    try:
        # Initialize mygene
        mg = mygene.MyGeneInfo()

        # Get gene info - use single query instead of querymany to avoid dup issues
        gene_info = mg.query(term, scopes="symbol", species="human", size=5)

        if not gene_info or "hits" not in gene_info or len(gene_info["hits"]) == 0:
            logger.warning(f"No gene info found for {term}")
            return None

        # Find the best gene ID - prefer NCBI gene IDs over Ensembl IDs
        gene_id = None
        gene_name = None

        for hit in gene_info["hits"]:
            if "_id" in hit:
                current_id = str(hit["_id"])
                current_name = hit.get("name", "")

                # Skip Ensembl IDs (start with ENSG) - prefer NCBI gene IDs (numeric)
                if current_id.startswith("ENSG"):
                    continue

                # Prefer numeric NCBI gene IDs
                if current_id.isdigit():
                    gene_id = current_id
                    gene_name = current_name
                    break

                # Fallback to any non-Ensembl ID
                if not gene_id:
                    gene_id = current_id
                    gene_name = current_name

        if not gene_id:
            logger.warning(
                f"No suitable gene ID found for {term} (only Ensembl IDs available)"
            )
            return None

        # Skip if we still got an Ensembl ID
        if gene_id.startswith("ENSG"):
            logger.warning(f"Only Ensembl ID available for {term}: {gene_id}")
            return None

        # Now use NCBI Entrez API with the reliable gene ID from mygene
        Entrez.email = description_config.email

        try:
            # Use esummary to get gene summary information
            summary_handle = Entrez.esummary(db="gene", id=gene_id)
            summary_results = Entrez.read(
                summary_handle, validate=False
            )  # Add validate=False
            summary_handle.close()

            if summary_results:
                # Handle both list and dict formats
                if isinstance(summary_results, list):
                    summary = summary_results[0] if summary_results else {}
                elif "DocumentSummarySet" in summary_results:
                    doc_summary = summary_results["DocumentSummarySet"][
                        "DocumentSummary"
                    ]
                    summary = (
                        doc_summary[0] if isinstance(doc_summary, list) else doc_summary
                    )
                else:
                    summary = summary_results

                description = ""

                # Try to extract description from various fields
                if isinstance(summary, dict):
                    if "Summary" in summary and summary["Summary"].strip():
                        description = str(summary["Summary"]).strip()
                    elif "Description" in summary and summary["Description"].strip():
                        description = str(summary["Description"]).strip()
                    elif "Name" in summary and summary["Name"].strip():
                        description = f"Gene: {summary['Name']}"

                # If we got a good description from API, use it
                if description and len(description) > 10:
                    return {
                        "description": description,
                        "definition_source": "NCBI Gene API (via mygene ID)",
                    }

        except Exception as api_error:
            logger.warning(
                f"NCBI API error for gene {term} (ID: {gene_id}): {api_error}"
            )

        # Fallback: use basic gene info from mygene
        if gene_name:
            basic_desc = f"{term}: {gene_name}"
            return {"description": basic_desc, "definition_source": "MyGene API"}

        return None

    except Exception as e:
        logger.warning(f"Error fetching Gene data for '{term}': {e}")
        return None


def fetch_from_books(term, description_config):
    """Fetch information from NCBI Bookshelf."""
    try:
        search_handle = Entrez.esearch(db="books", term=term, retmax=3)
        search_results = Entrez.read(search_handle)
        search_handle.close()

        if not search_results["IdList"]:
            return None

        book_id = search_results["IdList"][0]
        fetch_handle = Entrez.efetch(db="books", id=book_id, rettype="abstract")
        abstract_text = fetch_handle.read()
        fetch_handle.close()

        if abstract_text:
            # Clean up the text (remove XML tags, etc.)
            import re

            clean_text = re.sub(r"<[^>]+>", "", abstract_text)
            clean_text = clean_text.strip()[:500]  # Limit to 500 characters

            return (
                {"description": clean_text, "definition_source": "NCBI Books"}
                if clean_text
                else None
            )

        return None

    except Exception as e:
        print(f"Error fetching Books data for '{term}': {e}")
        return None


def create_dataset_description(term_type, df_pairs, df_descriptions):
    """
    Create a comprehensive description for the HuggingFace dataset.

    Args:
        term_type (str): Type of terms (cell_types, diseases, etc.)
        df_pairs (pd.DataFrame): Anchor-positive pairs
        df_descriptions (pd.DataFrame): Original descriptions

    Returns:
        str: Formatted dataset description
    """
    # Get 3 example pairs for better pattern recognition
    example_pairs = []
    if not df_pairs.empty:
        num_examples = min(3, len(df_pairs))
        for i in range(num_examples):
            row = df_pairs.iloc[i]
            example_pairs.append(
                {
                    "anchor": row["anchor"],
                    "positive": row["positive"][:150] + "..."
                    if len(row["positive"]) > 150
                    else row["positive"],
                    "source": row["source_database"],
                    "is_synonym": row.get("is_synonym", False),
                }
            )
    else:
        example_pairs = [
            {
                "anchor": "No examples available",
                "positive": "",
                "source": "",
                "is_synonym": False,
            }
        ]

    # Get unique sources and their citations
    sources_used = df_pairs["source_database"].unique()
    citations_section = ""

    for source in sources_used:
        citation_info = get_database_citation(source)
        citations_section += f"- **{citation_info['title']}**: {citation_info['citation']} [Link]({citation_info['link']})\n"

    # Count statistics
    total_pairs = len(df_pairs)
    main_terms = len(df_pairs[~df_pairs["is_synonym"]])  # Use ~ for Series negation
    synonym_pairs = len(df_pairs[df_pairs["is_synonym"]])

    # Create examples section
    examples_section = ""
    for i, pair in enumerate(example_pairs):
        synonym_indicator = " ✨ (synonym)" if pair["is_synonym"] else ""
        examples_section += f"""
**Example {i + 1}:**
```
Anchor: "{pair["anchor"]}"{synonym_indicator}
Positive: "{pair["positive"]}"
Source: {pair["source"]}
```
"""

    description = f"""# {term_type.replace("_", " ").title()} Anchor-Positive Pairs

This dataset contains anchor-positive pairs for training sentence transformers on biomedical {term_type.replace("_", " ")} terminology.

## 📊 Dataset Statistics
- **Total pairs**: {total_pairs:,}
- **Main term pairs**: {main_terms:,}
- **Synonym pairs**: {synonym_pairs:,}
- **Unique sources**: {len(sources_used)}

## 📝 Example Pairs

{examples_section}

## 🗂️ Dataset Structure
- **anchor**: The biomedical term (including synonyms)
- **positive**: Clean, authoritative definition from biomedical ontologies
- **source_database**: Database/ontology used (with proper attribution)
- **source_link**: Direct link to the ontology
- **is_synonym**: Boolean flag indicating if the anchor is a synonym

## 📚 Data Sources
All definitions are sourced from standardized biomedical ontologies via the EBI Ontology Lookup Service:

{citations_section}

## 🎯 Intended Use
This dataset is designed for:
- Training sentence transformers on biomedical terminology
- Fine-tuning models for biomedical text understanding
- Creating embeddings that capture semantic relationships between terms and definitions

## ⚖️ Licensing
Data is sourced from publicly available biomedical ontologies. Please cite the original ontologies when using this dataset.

## 🔄 Generation
Generated using ConCellT (Contextualized Cell Type Representations) - a framework for biomedical term description extraction and processing.
"""

    return description


def get_database_citation(source):
    """
    Get proper citation and link for database sources.

    Args:
        source (str): Source database name

    Returns:
        dict: Citation information with title, link, and citation
    """
    citations = {
        "CL Ontology": {
            "title": "Cell Ontology (CL)",
            "link": "https://www.ebi.ac.uk/ols/ontologies/cl",
            "citation": "Diehl, A.D., et al. (2016). The Cell Ontology 2016: enhanced content, modularization, and ontology interoperability. Journal of Biomedical Semantics, 7(1), 44.",
        },
        "MONDO Ontology": {
            "title": "Mondo Disease Ontology (MONDO)",
            "link": "https://www.ebi.ac.uk/ols/ontologies/mondo",
            "citation": "Vasilevsky, N.A., et al. (2022). Mondo: Unifying diseases for the world, by the world. medRxiv.",
        },
        "HP Ontology": {
            "title": "Human Phenotype Ontology (HPO)",
            "link": "https://www.ebi.ac.uk/ols/ontologies/hp",
            "citation": "Köhler, S., et al. (2021). The Human Phenotype Ontology in 2021. Nucleic Acids Research, 49(D1), D1207-D1217.",
        },
        "UBERON Ontology": {
            "title": "Uberon Anatomy Ontology",
            "link": "https://www.ebi.ac.uk/ols/ontologies/uberon",
            "citation": "Mungall, C.J., et al. (2012). Uberon, an integrative multi-species anatomy ontology. Genome Biology, 13(1), R5.",
        },
        "GO Ontology": {
            "title": "Gene Ontology (GO)",
            "link": "https://www.ebi.ac.uk/ols/ontologies/go",
            "citation": "The Gene Ontology Consortium (2021). The Gene Ontology resource: enriching a GOld mine. Nucleic Acids Research, 49(D1), D325-D334.",
        },
        "NCBITAXON Ontology": {
            "title": "NCBI Taxonomy",
            "link": "https://www.ebi.ac.uk/ols/ontologies/ncbitaxon",
            "citation": "Schoch, C.L., et al. (2020). NCBI Taxonomy: a comprehensive update on curation, resources and tools. Database, 2020, baaa062.",
        },
        "DOID Ontology": {
            "title": "Disease Ontology (DOID)",
            "link": "https://www.ebi.ac.uk/ols/ontologies/doid",
            "citation": "Schriml, L.M., et al. (2019). Human Disease Ontology 2018 update: classification, content and workflow expansion. Nucleic Acids Research, 47(D1), D955-D962.",
        },
        "FMA Ontology": {
            "title": "Foundational Model of Anatomy (FMA)",
            "link": "https://www.ebi.ac.uk/ols/ontologies/fma",
            "citation": "Rosse, C. & Mejino Jr, J.L. (2003). A reference ontology for biomedical informatics: the Foundational Model of Anatomy. Journal of Biomedical Informatics, 36(6), 478-500.",
        },
        "PR Ontology": {
            "title": "Protein Ontology (PRO)",
            "link": "https://www.ebi.ac.uk/ols/ontologies/pr",
            "citation": "Natale, D.A., et al. (2011). Protein Ontology: a controlled structured network of protein entities. Nucleic Acids Research, 39(suppl_1), D464-D467.",
        },
    }

    # Default for unknown sources
    default = {
        "title": source,
        "link": "https://www.ebi.ac.uk/ols/",
        "citation": f"Source: {source}. Retrieved from EBI Ontology Lookup Service.",
    }

    return citations.get(source, default)


def create_anchor_positive_pairs(df_descriptions, term_type):
    """
    Create anchor-positive pairs from term descriptions for sentence transformer training.

    Args:
        df_descriptions (pd.DataFrame): DataFrame with term descriptions
        term_type (str): Type of terms (for naming)

    Returns:
        pd.DataFrame: DataFrame with anchor-positive pairs and source information
    """
    pairs = []

    for _, row in df_descriptions.iterrows():
        # Handle NaN and empty descriptions properly
        if (
            pd.notna(row["description"]) and str(row["description"]).strip()
        ):  # Only create pairs if we have a description
            source = row.get("definition_source", "Unknown")
            citation_info = get_database_citation(source)

            pair = {
                "anchor": row["term"],
                "positive": row["description"],
                "source_database": citation_info["title"],
                "source_link": citation_info["link"],
            }

            # Add synonyms as additional anchor-positive pairs if available
            if row.get("synonyms") and row["synonyms"].strip():
                synonyms = [s.strip() for s in row["synonyms"].split(";") if s.strip()]
                for synonym in synonyms[
                    :3
                ]:  # Limit to 3 synonyms to avoid too many pairs
                    synonym_pair = pair.copy()
                    synonym_pair["anchor"] = synonym
                    synonym_pair["is_synonym"] = True
                    pairs.append(synonym_pair)

            # Add the main term pair
            pair["is_synonym"] = False
            pairs.append(pair)

    return pd.DataFrame(pairs)


def gen_term_descriptions(description_config):
    """
    Generate descriptions for various types of biomedical terms.

    This function fetches descriptions for cell types, diseases, tissues, organisms,
    and genes from NCBI databases (primarily MeSH) and saves them in separate CSV files.
    Optionally creates HuggingFace datasets with anchor-positive pairs.

    Args:
        description_config: TermDescriptionConfig object with search parameters
    """
    config = description_config.config

    # Define term types and their corresponding lists
    term_mappings = {
        "cell_types": description_config.cell_types,
        "diseases": description_config.diseases,
        "tissues": description_config.tissues,
        "organisms": description_config.organisms,
        "genes": description_config.genes,
    }

    # Store datasets for HuggingFace upload if requested
    hf_datasets = {}

    for term_type, terms_list in term_mappings.items():
        # Only process if we have terms provided
        if terms_list:
            print(f"\n📚 Processing {len(terms_list)} {term_type}...")

            # Fetch descriptions for this term type
            df_descriptions = fetch_term_descriptions(
                terms_list, term_type, description_config
            )

            # Filter out entries without descriptions before saving
            original_count = len(df_descriptions)
            df_descriptions_filtered = df_descriptions[
                df_descriptions["description"].notna()
                & (df_descriptions["description"].str.strip() != "")
            ].copy()

            # Log missing descriptions
            missing_count = original_count - len(df_descriptions_filtered)
            if missing_count > 0:
                missing_terms = df_descriptions[
                    df_descriptions["description"].isna()
                    | (df_descriptions["description"].str.strip() == "")
                ]["term"].tolist()
                print(
                    f"⚠️  {missing_count} {term_type} without descriptions (excluded from CSV):"
                )
                for i, term in enumerate(missing_terms[:10]):  # Show first 10
                    print(f"   • {term}")
                if len(missing_terms) > 10:
                    print(f"   • ... and {len(missing_terms) - 10} more")

            # Create subfolder for this term type
            subfolder = Path("descriptions") / term_type

            # Generate filename
            dataframe_name = f"{term_type}_descriptions.csv"

            # Save filtered CSV (only entries with descriptions)
            save_to_csv(
                dataframe=df_descriptions_filtered,
                dataframe_name=dataframe_name,
                config=description_config,
                subfolder=subfolder,
            )

            print(
                f"✅ Saved {len(df_descriptions_filtered)} {term_type} descriptions (filtered from {original_count} total)"
            )

            # Create anchor-positive pairs for HuggingFace if requested
            if config.save_to_hf:
                df_pairs = create_anchor_positive_pairs(
                    df_descriptions_filtered, term_type
                )
                if not df_pairs.empty:
                    # Create HuggingFace dataset
                    hf_dataset = Dataset.from_pandas(df_pairs)
                    dataset_name = f"{term_type}"
                    hf_datasets[dataset_name] = {
                        "dataset": hf_dataset,
                        "df_pairs": df_pairs,
                        "df_descriptions": df_descriptions_filtered,
                    }
                    print(
                        f"📦 Created {len(df_pairs)} anchor-positive pairs for {term_type}"
                    )

            # Print sample results (only from successfully fetched descriptions)
            if not df_descriptions_filtered.empty:
                print(f"Sample descriptions for {term_type}:")
                for _, row in df_descriptions_filtered.head(3).iterrows():
                    desc_preview = (
                        row["description"][:100] + "..."
                        if len(row["description"]) > 100
                        else row["description"]
                    )
                    print(f"  • {row['term']}: {desc_preview}")
                print()

    # Upload to HuggingFace Hub if requested
    if config.save_to_hf and hf_datasets:
        print(f"\n🚀 Uploading {len(hf_datasets)} datasets to HuggingFace Hub...")

        base_repo_name = f"{config.data_dir}"

        for dataset_name, dataset_info in hf_datasets.items():
            full_repo_name = f"{base_repo_name}_{dataset_name}"

            print(f"📤 Uploading {full_repo_name}...")

            try:
                # Get the dataset and create description
                dataset = dataset_info["dataset"]
                df_pairs = dataset_info["df_pairs"]
                df_descriptions = dataset_info["df_descriptions"]

                # Extract term type from dataset name
                term_type = dataset_name.replace("_anchor_positive", "")
                description = create_dataset_description(
                    term_type, df_pairs, df_descriptions
                )

                # First, ensure the repository exists and upload dataset
                from huggingface_hub import HfApi
                import tempfile
                import os
                import time

                api = HfApi()

                try:
                    # Create repository first (this will not fail if it already exists)
                    api.create_repo(
                        repo_id=full_repo_name,
                        repo_type="dataset",
                        private=True,
                        token=hf_config.HF_TOKEN_UPLOAD,
                        exist_ok=True,  # Don't fail if repo already exists
                    )
                    print(f"📁 Repository created/verified: {full_repo_name}")
                except Exception as create_error:
                    print(f"⚠️  Repository creation warning: {create_error}")

                # Upload dataset
                dataset.push_to_hub(
                    full_repo_name, private=True, token=hf_config.HF_TOKEN_UPLOAD
                )

                # Small delay to ensure repository is fully available
                time.sleep(5)

                # Create and upload README.md file
                try:
                    # Get user info to construct full repo path (username/repo_name)
                    user_info = api.whoami(token=hf_config.HF_TOKEN_UPLOAD)
                    username = user_info["name"]
                    full_repo_path = f"{username}/{full_repo_name}"

                    print(f"📁 Uploading README to: {full_repo_path}")

                    # Create temporary README.md file
                    with tempfile.NamedTemporaryFile(
                        mode="w", suffix=".md", delete=False, encoding="utf-8"
                    ) as f:
                        f.write(description)
                        temp_readme_path = f.name

                    # Upload README.md to the repository with full path
                    api.upload_file(
                        path_or_fileobj=temp_readme_path,
                        path_in_repo="README.md",
                        repo_id=full_repo_path,
                        repo_type="dataset",
                        token=hf_config.HF_TOKEN_UPLOAD,
                        commit_message="Add comprehensive dataset documentation",
                    )

                    # Clean up temporary file
                    os.unlink(temp_readme_path)

                    print(
                        f"✅ Successfully uploaded with README: https://huggingface.co/datasets/{full_repo_path}"
                    )

                except Exception as desc_error:
                    print(
                        f"✅ Dataset uploaded: https://huggingface.co/datasets/{full_repo_name}"
                    )
                    print(f"⚠️  Could not add README automatically: {desc_error}")
                    print(
                        "💡 You can manually copy the description from the terminal output above"
                    )

                    # Print the description so user can copy it manually if needed
                    print("\n" + "=" * 60)
                    print("📋 README CONTENT (for manual copy if needed):")
                    print("=" * 60)
                    print(description)
                    print("=" * 60)

            except Exception as e:
                print(f"❌ Failed to upload {full_repo_name}: {e}")

        print("🎉 HuggingFace upload complete!")

    print("🎉 Term description fetching complete!")
    print(f"📁 Check results in: out/data/{config.data_dir}/descriptions/")

    return hf_datasets if config.save_to_hf else None
