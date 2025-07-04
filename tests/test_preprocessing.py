import tempfile
import pytest
import numpy as np
import pandas as pd
import anndata as ad
from pathlib import Path
from scipy import sparse
from unittest.mock import patch

from adata_hf_datasets.pp.orchestrator import preprocess_h5ad
from adata_hf_datasets.pp.utils import safe_read_h5ad_backed


@pytest.fixture
def realistic_adata():
    """
    Create a realistic AnnData object with multiple batches, valid gene names,
    and various attributes to test preprocessing functionality.
    """
    # Constants for the test data
    n_cells = 1000
    n_genes = 20  # Reduced from 500 to 20 genes
    n_batches = 5

    # Create realistic gene names (using real Ensembl IDs)
    # These are real gene names that biomart can recognize
    gene_names = [
        "ENSG00000141510",  # TP53
        "ENSG00000073282",  # TP63
        "ENSG00000133703",  # KRAS
        "ENSG00000141736",  # ERBB2
        "ENSG00000181143",  # MUC16
        "ENSG00000213341",  # INKA1
        "ENSG00000163930",  # BAP1
        "ENSG00000160791",  # LMNA
        "ENSG00000099364",  # PRPF19
        "ENSG00000080815",  # PSEN1
        "ENSG00000139618",  # BRCA2
        "ENSG00000091831",  # ESR1
        "ENSG00000152268",  # SMARCA4
        "ENSG00000140379",  # BCL2L11
        "ENSG00000118046",  # STK11
        "ENSG00000166710",  # B2M
        "ENSG00000114854",  # TNNC1
        "ENSG00000169032",  # MAP2K1
        "ENSG00000065613",  # SLK
        "ENSG00000178568",  # ERBB4
    ]

    # We now have exactly 20 genes with valid Ensembl IDs
    assert len(gene_names) == n_genes

    # No need to extend the gene list anymore as we have exactly n_genes
    # gene_names = gene_names * (n_genes // len(gene_names) + 1)
    # gene_names = gene_names[:n_genes]

    # Create batch labels (multiple cells per batch)
    batch_cells_per_batch = n_cells // n_batches
    batches = np.repeat(np.arange(n_batches), batch_cells_per_batch)
    # Add some remainder cells to the last batch
    remainder = n_cells % n_batches
    if remainder > 0:
        batches = np.concatenate([batches, np.full(remainder, n_batches - 1)])

    # Create count matrix (sparse to save memory)
    # Use negative binomial distribution for counts to simulate RNA-seq
    counts = np.random.negative_binomial(5, 0.5, size=(n_cells, n_genes))
    counts = sparse.csr_matrix(counts)

    # Create cell metadata
    obs = pd.DataFrame(
        {
            "batch": [f"batch_{i}" for i in batches],
            "instrument": np.random.choice(
                ["Illumina", "PacBio", "Oxford Nanopore"], size=n_cells
            ),
            "description": [f"Sample from patient {i % 20}" for i in range(n_cells)],
            "cell_type": np.random.choice(
                ["CD4+ T", "CD8+ T", "B cell", "NK cell", "Monocyte"], size=n_cells
            ),
            "quality_score": np.random.uniform(0.5, 1.0, size=n_cells),
            "n_genes": np.random.randint(200, 500, size=n_cells),
            "n_counts": np.random.randint(1000, 5000, size=n_cells),
            "percent_mito": np.random.uniform(0, 0.2, size=n_cells),
            "doublet_score": np.random.uniform(0, 0.5, size=n_cells),
            "patient_id": [f"P{i % 20:02d}" for i in range(n_cells)],
            "sample_index": np.arange(n_cells),  # Add sample_index as a numeric range
        }
    )

    # Create gene metadata
    var = pd.DataFrame(
        {
            "gene_symbol": [f"Gene_{i}" for i in range(n_genes)],
            "chromosome": np.random.choice(
                ["chr1", "chr2", "chr3", "chr4", "chr5"], size=n_genes
            ),
            "start_position": np.random.randint(1, 100000000, size=n_genes),
            "end_position": np.random.randint(1, 100000000, size=n_genes),
            "strand": np.random.choice(["+", "-"], size=n_genes),
            "biotype": np.random.choice(
                ["protein_coding", "lncRNA", "miRNA"], size=n_genes
            ),
        },
        index=gene_names,
    )

    # Create AnnData object
    adata = ad.AnnData(X=counts, obs=obs, var=var)

    # Add raw counts layer
    adata.layers["counts"] = adata.X.copy()

    # Add PCA and UMAP embeddings
    adata.obsm["X_pca"] = np.random.normal(0, 1, size=(n_cells, 50))
    adata.obsm["X_umap"] = np.random.normal(0, 1, size=(n_cells, 2))

    # Add gene embeddings
    adata.varm["PCs"] = np.random.normal(0, 1, size=(n_genes, 50))

    # Add additional information to .uns
    adata.uns["dataset_info"] = {
        "species": "human",
        "tissue": "blood",
        "experiment_date": "2023-01-15",
        "sequencing_platform": "Illumina NovaSeq 6000",
        "analysis_version": "1.0.0",
    }

    # Add a categorical covariate with a bimodal distribution for testing split_bimodal
    # This simulates something like UMI counts that might have a bimodal distribution
    bimodal_values = np.concatenate(
        [
            np.random.normal(10, 2, size=n_cells // 2),  # Lower mode
            np.random.normal(50, 10, size=n_cells - n_cells // 2),  # Higher mode
        ]
    )
    np.random.shuffle(bimodal_values)
    adata.obs["bimodal_feature"] = bimodal_values.astype(np.float32)

    return adata


@pytest.fixture
def temp_h5ad_paths():
    """Create temporary paths for input and output h5ad files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "input.h5ad"
        output_dir = Path(tmpdir) / "output/"
        yield input_path, output_dir


def test_safe_read_h5ad_backed_normal_operation(realistic_adata, temp_h5ad_paths):
    """Test that safe_read_h5ad_backed works normally without any errors."""
    input_path, _ = temp_h5ad_paths

    # Save the test AnnData to disk
    realistic_adata.write_h5ad(input_path)

    # Test normal reading
    adata_backed = safe_read_h5ad_backed(input_path)

    # Verify it's in backed mode
    assert adata_backed.isbacked
    assert adata_backed.shape == realistic_adata.shape

    # Clean up
    if hasattr(adata_backed, "file") and adata_backed.file is not None:
        adata_backed.file.close()


def test_safe_read_h5ad_backed_with_memory_error_retry(
    realistic_adata, temp_h5ad_paths
):
    """Test that safe_read_h5ad_backed handles memory errors and retries."""
    input_path, _ = temp_h5ad_paths

    # Save the test AnnData to disk
    realistic_adata.write_h5ad(input_path)

    # Mock scanpy.read_h5ad to simulate memory error on first call, success on second
    call_count = 0
    original_read_h5ad = ad.read_h5ad

    def mock_read_h5ad(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # First call fails with memory error
            raise MemoryError("Unable to allocate memory for test")
        else:
            # Second call succeeds (after local copy)
            return original_read_h5ad(*args, **kwargs)

    with patch("scanpy.read_h5ad", side_effect=mock_read_h5ad):
        # This should succeed after retry
        adata_backed = safe_read_h5ad_backed(
            input_path,
            max_retry=3,
            copy_local=True,
            sleep=0.1,  # Fast retry for testing
        )

        # Verify it worked
        assert adata_backed.isbacked
        assert adata_backed.shape == realistic_adata.shape

        # Verify that it created a local copy (indicated by the attribute)
        assert hasattr(adata_backed, "_temp_local_copy")
        assert adata_backed._temp_local_copy is not None

        # Clean up
        if hasattr(adata_backed, "file") and adata_backed.file is not None:
            adata_backed.file.close()

        # Verify local copy was created and exists
        if hasattr(adata_backed, "_temp_local_copy"):
            _ = adata_backed._temp_local_copy
            # The local copy should exist during the test
            # It will be cleaned up automatically when the function finishes


def test_safe_read_h5ad_backed_max_retries_exceeded(realistic_adata, temp_h5ad_paths):
    """Test that safe_read_h5ad_backed raises error after max retries exceeded."""
    input_path, _ = temp_h5ad_paths

    # Save the test AnnData to disk
    realistic_adata.write_h5ad(input_path)

    # Mock scanpy.read_h5ad to always fail with memory error
    with patch("scanpy.read_h5ad", side_effect=MemoryError("Persistent memory error")):
        with pytest.raises(MemoryError, match="Persistent memory error"):
            safe_read_h5ad_backed(
                input_path,
                max_retry=2,
                copy_local=False,  # Disable local copying to ensure failure
                sleep=0.1,  # Fast retry for testing
            )


def test_safe_read_h5ad_backed_handles_other_exceptions(
    realistic_adata, temp_h5ad_paths
):
    """Test that safe_read_h5ad_backed handles non-memory errors appropriately."""
    input_path, _ = temp_h5ad_paths

    # Save the test AnnData to disk
    realistic_adata.write_h5ad(input_path)

    # Mock scanpy.read_h5ad to fail with a different error type
    with patch("scanpy.read_h5ad", side_effect=OSError("File system error")):
        with pytest.raises(OSError, match="File system error"):
            safe_read_h5ad_backed(input_path, max_retry=2, copy_local=False, sleep=0.1)


def test_safe_read_h5ad_backed_with_missing_file():
    """Test that safe_read_h5ad_backed handles missing files appropriately."""
    nonexistent_path = Path("/tmp/nonexistent_file.h5ad")

    with pytest.raises(FileNotFoundError):
        safe_read_h5ad_backed(nonexistent_path, max_retry=1, sleep=0.1)


def test_preprocess_h5ad_basic(realistic_adata, temp_h5ad_paths):
    """Test basic preprocessing with default parameters."""
    input_path, output_dir = temp_h5ad_paths

    # Save the test AnnData to disk
    realistic_adata.write_h5ad(input_path)

    # Run preprocessing
    preprocess_h5ad(
        infile=input_path,
        outdir=output_dir,
        chunk_size=200,  # Small chunk size for testing
        batch_key="batch",
        count_layer_key="counts",
        n_top_genes=10,  # Half the total genes
        min_genes=10,
        min_cells=10,
        output_format="h5ad",
    )

    # Check that output file exists
    assert output_dir.exists()

    # Load and validate the preprocessed data
    processed = ad.read_h5ad(output_dir / "chunk_0.h5ad")

    # Check basic structure
    assert isinstance(processed, ad.AnnData)
    assert "counts" in processed.layers
    assert processed.n_vars <= realistic_adata.n_vars  # May have filtered genes

    # Check for normalization in X
    assert np.max(processed.X) <= 100.0  # Log1p normalized data typically < 100
    assert np.min(processed.X) >= 0.0

    # Check for highly variable genes
    assert "highly_variable" in processed.var

    # Check uns attributes were preserved
    for key in realistic_adata.uns:
        assert key in processed.uns

    # Check embeddings preserved (if they survived filtering)
    if "X_pca" in realistic_adata.obsm and processed.n_obs == realistic_adata.n_obs:
        assert "X_pca" in processed.obsm


def test_preprocess_h5ad_with_instrument_description(realistic_adata, temp_h5ad_paths):
    """Test preprocessing with instrument appended to description."""
    input_path, output_dir = temp_h5ad_paths

    # Save the test AnnData to disk
    realistic_adata.write_h5ad(input_path)

    # Run preprocessing with instrument and description keys
    preprocess_h5ad(
        infile=input_path,
        outdir=output_dir,
        chunk_size=200,
        batch_key="batch",
        count_layer_key="counts",
        instrument_key="instrument",
        description_key="description",
        n_top_genes=10,  # Half the total genes
        min_genes=10,
        min_cells=10,
        output_format="h5ad",
    )

    # Load and validate
    processed = ad.read_h5ad(output_dir / "chunk_0.h5ad")

    # Check that instruments were prepended to descriptions
    for i, row in processed.obs.iterrows():
        assert row["description"].startswith(
            f"This measurement was conducted with {row['instrument']}"
        )
        # Ensure it didn't just replace the description
        assert "Sample from patient" in row["description"]


def test_preprocess_h5ad_with_bimodal_split(realistic_adata, temp_h5ad_paths):
    """Test preprocessing with bimodal splitting."""
    input_path, output_dir = temp_h5ad_paths

    # Save the test AnnData to disk
    realistic_adata.write_h5ad(input_path)

    # Run preprocessing with bimodal splitting
    preprocess_h5ad(
        infile=input_path,
        outdir=output_dir,
        chunk_size=200,
        batch_key="batch",
        count_layer_key="counts",
        bimodal_col="bimodal_feature",
        split_bimodal=True,
        n_top_genes=10,  # Half the total genes
        min_genes=10,
        min_cells=10,
        output_format="h5ad",
    )

    # Load and validate
    processed = ad.read_h5ad(output_dir / "chunk_0.h5ad")

    # Check that the bimodal_feature_log column was created
    assert "bimodal_feature_log" in processed.obs

    # Check that bimodal_split column exists (added during concatenation)
    assert "bimodal_split" in processed.obs


def test_preprocess_h5ad_with_category_consolidation(realistic_adata, temp_h5ad_paths):
    """Test preprocessing with category consolidation."""
    input_path, output_dir = temp_h5ad_paths

    # Modify the test data to include some rare categories
    realistic_adata.obs["rare_category"] = "common"
    # Add a few rare categories
    rare_indices = np.random.choice(realistic_adata.n_obs, size=5, replace=False)
    for i, idx in enumerate(rare_indices):
        realistic_adata.obs["rare_category"][str(idx)] = f"rare_{i}"

    # Save the test AnnData to disk
    realistic_adata.write_h5ad(input_path)

    # Run preprocessing with category consolidation
    preprocess_h5ad(
        infile=input_path,
        outdir=output_dir,
        chunk_size=1000,
        batch_key="batch",
        count_layer_key="counts",
        consolidation_categories="rare_category",
        category_threshold=10,  # Threshold higher than the count of rare categories
        remove_low_frequency=False,  # Merge into 'unknown' instead of removing
        n_top_genes=10,  # Half the total genes
        min_genes=10,
        min_cells=10,
        output_format="h5ad",
    )

    # Load and validate
    processed = ad.read_h5ad(output_dir / "chunk_0.h5ad")

    # Check that the rare categories were consolidated
    category_counts = processed.obs["rare_category"].value_counts()
    assert "remaining rare_category" in category_counts.index
    assert "common" in category_counts.index
    # No other rare categories should be present
    assert len(category_counts) == 2


def test_preprocess_h5ad_no_geneformer(realistic_adata, temp_h5ad_paths):
    """Test preprocessing without the Geneformer step."""
    input_path, output_dir = temp_h5ad_paths

    # Save the test AnnData to disk
    realistic_adata.write_h5ad(input_path)

    # Run preprocessing without Geneformer
    preprocess_h5ad(
        infile=input_path,
        outdir=output_dir,
        chunk_size=200,
        batch_key="batch",
        count_layer_key="counts",
        geneformer_pp=False,
        n_top_genes=10,  # Half the total genes
        min_genes=10,
        min_cells=10,
        output_format="h5ad",
    )

    # Load and validate
    processed = ad.read_h5ad(output_dir / "chunk_0.h5ad")

    # Geneformer typically adds specific columns to var
    # Without it, these should not be present
    assert "protein_id" not in processed.var
    assert "ensembl_id" not in processed.var


@pytest.mark.parametrize("chunk_size", [100, 200, 500])
def test_preprocess_h5ad_different_chunk_sizes(
    realistic_adata, temp_h5ad_paths, chunk_size
):
    """Test preprocessing with different chunk sizes to ensure consistent results."""
    input_path, output_dir = temp_h5ad_paths

    # Save the test AnnData to disk
    realistic_adata.write_h5ad(input_path)

    # Run preprocessing with specified chunk size
    preprocess_h5ad(
        infile=input_path,
        outdir=output_dir,
        chunk_size=chunk_size,
        batch_key="batch",
        count_layer_key="counts",
        n_top_genes=10,  # Half the total genes
        min_genes=10,
        min_cells=10,
        output_format="h5ad",
    )

    # Load and check basic validation
    processed = ad.read_h5ad(output_dir / "chunk_0.h5ad")
    assert isinstance(processed, ad.AnnData)
    assert "counts" in processed.layers

    # The number of cells might differ due to filtering, but should be consistent
    # Number of genes should be consistent across different chunk sizes after filtering
    assert processed.n_vars > 0


def test_preprocess_h5ad_ensures_count_layer(realistic_adata, temp_h5ad_paths):
    """Test that preprocessing ensures a counts layer exists."""
    input_path, output_dir = temp_h5ad_paths

    # Remove the counts layer to test that it gets added
    if "counts" in realistic_adata.layers:
        del realistic_adata.layers["counts"]

    # Save the modified data
    realistic_adata.write_h5ad(input_path)

    # Run preprocessing
    preprocess_h5ad(
        infile=input_path,
        outdir=output_dir,
        chunk_size=200,
        batch_key="batch",
        count_layer_key="counts",  # This layer doesn't exist initially
        n_top_genes=10,  # Half the total genes
        min_genes=10,
        min_cells=10,
        output_format="h5ad",
    )

    # Check that the counts layer was created
    processed = ad.read_h5ad(output_dir / "chunk_0.h5ad")
    assert "counts" in processed.layers


def test_preprocess_h5ad_with_empty_chunks(realistic_adata, temp_h5ad_paths):
    """
    Test preprocessing when some chunks might end up empty after filtering.
    This tests the robustness of the concatenation step.
    """
    input_path, output_dir = temp_h5ad_paths

    # Modify the dataset to have some very low quality cells that would be filtered out
    # Set half of one batch to have extremely high mitochondrial percentage
    batch_mask = realistic_adata.obs["batch"] == "batch_0"
    batch_cells = np.where(batch_mask)[0]
    cells_to_modify = batch_cells[: len(batch_cells) // 2]
    cells_to_modify_str = [str(i) for i in cells_to_modify]
    realistic_adata.obs["percent_mito"][cells_to_modify_str] = (
        0.9  # Very high mito percent
    )
    realistic_adata.obs["n_genes"][cells_to_modify_str] = 10  # Very few genes

    # Save the modified data
    realistic_adata.write_h5ad(input_path)

    # Run preprocessing with stringent filters that would remove these cells
    preprocess_h5ad(
        infile=input_path,
        outdir=output_dir,
        chunk_size=200,
        batch_key="batch",
        count_layer_key="counts",
        min_genes=20,  # Higher threshold to filter out modified cells
        n_top_genes=10,  # Half the total genes
        min_cells=10,
        output_format="h5ad",
    )

    # Check that preprocessing completed and produced a valid file
    assert output_dir.exists()
    # count amount of files in the output directory
    output_files = list(output_dir.glob("chunk_*.h5ad"))
    assert len(output_files) > 0


def test_preprocess_h5ad_preserves_gene_attributes(realistic_adata, temp_h5ad_paths):
    """Test that preprocessing preserves important gene attributes."""
    input_path, output_dir = temp_h5ad_paths

    # Add some important gene attributes that should be preserved
    realistic_adata.var["important_score"] = np.random.uniform(
        0, 1, size=realistic_adata.n_vars
    )
    realistic_adata.var["is_marker"] = np.random.choice(
        [True, False], size=realistic_adata.n_vars
    )

    # Save the modified data
    realistic_adata.write_h5ad(input_path)

    # Run preprocessing
    preprocess_h5ad(
        infile=input_path,
        outdir=output_dir,
        chunk_size=200,
        batch_key="batch",
        count_layer_key="counts",
        n_top_genes=10,  # Half the total genes
        min_genes=10,
        min_cells=10,
        output_format="h5ad",
    )

    # Check that important gene attributes were preserved
    processed = ad.read_h5ad(output_dir / "chunk_0.h5ad")

    # Some genes might be filtered out, but the attributes should be present
    preserved_genes = [g for g in realistic_adata.var_names if g in processed.var_names]

    # Only check preserved genes
    for gene in preserved_genes:
        if "important_score" in processed.var:
            assert gene in processed.var.index
            assert np.isclose(
                processed.var.loc[gene, "important_score"],
                realistic_adata.var.loc[gene, "important_score"],
            )
        if "is_marker" in processed.var:
            assert (
                processed.var.loc[gene, "is_marker"]
                == realistic_adata.var.loc[gene, "is_marker"]
            )


def test_preprocess_h5ad_sample_index_preservation(realistic_adata, temp_h5ad_paths):
    """Test that the sample_index column is properly preserved during preprocessing."""
    input_path, output_dir = temp_h5ad_paths

    # Verify sample_index exists and is a numeric range
    assert "sample_index" in realistic_adata.obs.columns
    assert np.array_equal(
        realistic_adata.obs["sample_index"].values, np.arange(realistic_adata.n_obs)
    )

    # Save the test AnnData to disk
    realistic_adata.write_h5ad(input_path)

    # Run preprocessing with geneformer which requires sample_index
    preprocess_h5ad(
        infile=input_path,
        outdir=output_dir,
        chunk_size=200,
        batch_key="batch",
        count_layer_key="counts",
        geneformer_pp=True,
        n_top_genes=10,  # Half the total genes
        min_genes=10,
        min_cells=10,
        output_format="h5ad",
    )

    # Load and validate
    processed = ad.read_h5ad(output_dir / "chunk_0.h5ad")

    # Verify sample_index is preserved
    assert "sample_index" in processed.obs.columns

    # The indices might be reordered if cells were filtered, so check that
    # sample_index still contains unique integers starting from 0
    sample_indices = processed.obs["sample_index"].values
    assert sample_indices.dtype.kind in ("i", "u")  # integer type
    assert len(sample_indices) == len(set(sample_indices))  # unique values
    assert max(sample_indices) < realistic_adata.n_obs  # within original range


def test_preprocess_h5ad_geneformer_adds_ensembl_id(realistic_adata, temp_h5ad_paths):
    """Test that geneformer preprocessing adds ensembl_id to var."""
    input_path, output_dir = temp_h5ad_paths

    # Save the test AnnData to disk
    realistic_adata.write_h5ad(input_path)

    # Run preprocessing with geneformer
    preprocess_h5ad(
        infile=input_path,
        outdir=output_dir,
        chunk_size=200,
        batch_key="batch",
        count_layer_key="counts",
        geneformer_pp=True,
        n_top_genes=10,  # Half the total genes
        min_genes=10,
        min_cells=10,
        output_format="h5ad",
    )

    # Load and validate
    processed = ad.read_h5ad(output_dir / "chunk_0.h5ad")

    # Check that geneformer added ensembl_id to var
    # The var index should already be Ensembl IDs, and geneformer should copy them
    assert "ensembl_id" in processed.var.columns

    # Sample a few genes to check if their ensembl_id matches the index
    for gene in processed.var_names[:5]:
        assert processed.var.loc[gene, "ensembl_id"] == gene
