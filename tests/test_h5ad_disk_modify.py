import os
import tempfile
import pytest
import numpy as np
import anndata as ad
import h5py
from pathlib import Path
import gc
from concurrent.futures import ThreadPoolExecutor

from adata_hf_datasets.file_utils import (
    add_obs_column_to_h5ad,
    add_sample_index_to_h5ad,
)


@pytest.fixture
def temp_h5ad_file():
    """Create a temporary h5ad file for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a simple AnnData object
        n_obs = 100
        n_vars = 50
        X = np.random.rand(n_obs, n_vars)
        adata = ad.AnnData(X=X)

        # Save to temporary file
        temp_path = Path(tmpdir) / "test.h5ad"
        adata.write_h5ad(temp_path)

        yield temp_path


def test_add_sample_index_to_h5ad(temp_h5ad_file):
    """Test the legacy function that adds sample_index to h5ad file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "out.h5ad"

        # Apply the function
        result_path = add_sample_index_to_h5ad(temp_h5ad_file, out_path)

        # Verify the result path is correct
        assert result_path == out_path
        assert result_path.exists()

        # Load and verify the modified file
        adata = ad.read_h5ad(result_path)
        assert "sample_index" in adata.obs.columns

        # Verify the sample_index values
        n_obs = adata.n_obs
        expected_indices = np.arange(n_obs)
        np.testing.assert_array_equal(adata.obs["sample_index"], expected_indices)


def test_add_obs_column_to_h5ad_default(temp_h5ad_file):
    """Test add_obs_column_to_h5ad with default parameters (should behave like add_sample_index_to_h5ad)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "out.h5ad"

        # Apply the function
        result_path = add_obs_column_to_h5ad(temp_h5ad_file, out_path)

        # Load and verify the modified file
        adata = ad.read_h5ad(result_path)
        assert "sample_index" in adata.obs.columns

        # Verify the sample_index values
        n_obs = adata.n_obs
        expected_indices = np.arange(n_obs)
        np.testing.assert_array_equal(adata.obs["sample_index"], expected_indices)


def test_add_obs_column_to_h5ad_custom_column(temp_h5ad_file):
    """Test add_obs_column_to_h5ad with a custom column name and data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "out.h5ad"

        # Create custom column data
        original_adata = ad.read_h5ad(temp_h5ad_file)
        n_obs = original_adata.n_obs
        custom_data = np.random.randint(0, 5, size=n_obs)

        # Apply the function
        result_path = add_obs_column_to_h5ad(
            temp_h5ad_file,
            out_path,
            column_name="cluster_id",
            column_data=custom_data,
            dtype=np.int32,
        )

        # Load and verify the modified file
        adata = ad.read_h5ad(result_path)
        assert "cluster_id" in adata.obs.columns

        # Verify the custom column values
        np.testing.assert_array_equal(adata.obs["cluster_id"], custom_data)


def test_add_obs_column_to_h5ad_string_data(temp_h5ad_file):
    """Test add_obs_column_to_h5ad with string data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "out.h5ad"

        # Create string column data
        original_adata = ad.read_h5ad(temp_h5ad_file)
        n_obs = original_adata.n_obs
        cell_types = np.array(["A", "B", "C", "D", "E"])
        custom_data = cell_types[np.random.randint(0, 5, size=n_obs)]

        # Apply the function with string data
        result_path = add_obs_column_to_h5ad(
            temp_h5ad_file,
            out_path,
            column_name="cell_type",
            column_data=custom_data,  # No need to convert to bytes anymore
            is_categorical=True,  # Mark as categorical
        )

        # Load and verify the modified file
        adata = ad.read_h5ad(result_path)
        assert "cell_type" in adata.obs.columns

        # Verify the string column values
        # Categorical data is already properly handled by pandas/anndata
        np.testing.assert_array_equal(
            adata.obs["cell_type"].astype(str).values, custom_data
        )


def test_add_obs_column_to_h5ad_error_handling(temp_h5ad_file):
    """Test error handling in add_obs_column_to_h5ad."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "out.h5ad"

        # Create incorrect size data
        original_adata = ad.read_h5ad(temp_h5ad_file)
        n_obs = original_adata.n_obs
        wrong_size_data = np.random.rand(n_obs + 10)  # Wrong size

        # Test missing data for custom column
        with pytest.raises(ValueError, match="column_data must be provided"):
            add_obs_column_to_h5ad(
                temp_h5ad_file,
                out_path,
                column_name="custom_column",  # Not sample_index, requires data
                column_data=None,
            )

        # Test wrong size data
        with pytest.raises(
            ValueError, match="column_data length .* doesn't match n_obs"
        ):
            add_obs_column_to_h5ad(
                temp_h5ad_file,
                out_path,
                column_name="custom_column",
                column_data=wrong_size_data,
            )


def test_add_multiple_columns_sequentially(temp_h5ad_file):
    """Test adding multiple columns sequentially."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create intermediate and final output paths
        temp_out = Path(tmpdir) / "intermediate.h5ad"
        final_out = Path(tmpdir) / "final.h5ad"

        # Read original file to get n_obs
        original_adata = ad.read_h5ad(temp_h5ad_file)
        n_obs = original_adata.n_obs

        # First add sample_index
        intermediate_path = add_obs_column_to_h5ad(temp_h5ad_file, temp_out)

        # Then add a custom column
        custom_data = np.random.random(n_obs)
        final_path = add_obs_column_to_h5ad(
            intermediate_path,
            final_out,
            column_name="random_values",
            column_data=custom_data,
            dtype=np.float64,
        )

        # Verify both columns exist
        adata = ad.read_h5ad(final_path)
        assert "sample_index" in adata.obs.columns
        assert "random_values" in adata.obs.columns

        # Verify the data
        np.testing.assert_array_equal(adata.obs["sample_index"], np.arange(n_obs))
        np.testing.assert_array_almost_equal(adata.obs["random_values"], custom_data)


def test_file_is_properly_closed(temp_h5ad_file):
    """Test that the file is properly closed after modification."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "out.h5ad"

        # Apply the function
        result_path = add_sample_index_to_h5ad(temp_h5ad_file, out_path)

        # Force garbage collection to close any lingering file handles
        gc.collect()

        # Try to open the file immediately after modification
        # This should succeed if the file was properly closed
        try:
            with h5py.File(result_path, "r+") as f:
                # Try to modify the file
                if "test_group" not in f:
                    f.create_group("test_group")
            success = True
        except Exception as e:
            success = False
            print(f"Failed to open file: {e}")

        assert success, "File was not properly closed after modification"


def test_file_accessible_after_modification(temp_h5ad_file):
    """Test that the file can be immediately accessed after modification."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "out.h5ad"

        # Apply the function
        result_path = add_sample_index_to_h5ad(temp_h5ad_file, out_path)

        # Try to read the file immediately after modification
        try:
            # Try to open with AnnData
            adata = ad.read_h5ad(result_path)
            assert "sample_index" in adata.obs.columns

            # Try to open with h5py
            with h5py.File(result_path, "r") as f:
                assert "obs" in f
                assert "sample_index" in f["obs"]

            success = True
        except Exception as e:
            success = False
            print(f"Failed to access file: {e}")

        assert success, "File was not immediately accessible after modification"


def test_multiple_read_operations(temp_h5ad_file):
    """Test that multiple read operations can be performed on the modified file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "out.h5ad"

        # Apply the function
        result_path = add_sample_index_to_h5ad(temp_h5ad_file, out_path)

        # Perform multiple read operations
        for _ in range(5):
            adata = ad.read_h5ad(result_path)
            assert "sample_index" in adata.obs.columns
            # Force close by deleting reference
            del adata
            gc.collect()


def _concurrent_worker(file_path, index):
    """Worker function for concurrent access test."""
    try:
        adata = ad.read_h5ad(file_path)
        # Do something with the file
        has_index = "sample_index" in adata.obs.columns
        del adata
        return has_index
    except Exception as e:
        print(f"Worker {index} failed: {e}")
        return False


def test_concurrent_file_access(temp_h5ad_file):
    """Test that multiple processes can access the file concurrently after modification."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "out.h5ad"

        # Apply the function
        result_path = add_sample_index_to_h5ad(temp_h5ad_file, out_path)

        # Try concurrent access from multiple threads
        num_workers = 5
        results = []

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(_concurrent_worker, result_path, i)
                for i in range(num_workers)
            ]
            for future in futures:
                results.append(future.result())

        # All workers should have successfully accessed the file
        assert all(results), "Concurrent file access test failed"


def test_large_string_categorical_data(temp_h5ad_file):
    """Test adding a large categorical string column."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "out.h5ad"

        # Create string column data with longer strings
        original_adata = ad.read_h5ad(temp_h5ad_file)
        n_obs = original_adata.n_obs

        # Create some longer category names (simulating cell type annotations)
        cell_types = np.array(
            [
                "CD4_T_cell_naive",
                "CD8_T_cell_effector_memory",
                "B_cell_naive",
                "Natural_killer_cell_cytotoxic",
                "Monocyte_classical",
            ]
        )
        custom_data = cell_types[np.random.randint(0, 5, size=n_obs)]

        # Apply the function
        result_path = add_obs_column_to_h5ad(
            temp_h5ad_file,
            out_path,
            column_name="detailed_cell_type",
            column_data=custom_data,
            is_categorical=True,
        )

        # Load and verify
        adata = ad.read_h5ad(result_path)
        assert "detailed_cell_type" in adata.obs.columns

        # Check data is preserved correctly including longer strings
        np.testing.assert_array_equal(
            adata.obs["detailed_cell_type"].astype(str).values, custom_data
        )


def test_add_column_to_existing_copy(temp_h5ad_file):
    """Test adding a column to a file that already has modifications."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create the initial copy with sample_index
        first_copy = Path(tmpdir) / "first_copy.h5ad"
        add_sample_index_to_h5ad(temp_h5ad_file, first_copy)

        # Now add another column to the same file (in-place)
        original_adata = ad.read_h5ad(temp_h5ad_file)
        n_obs = original_adata.n_obs
        custom_data = np.random.randint(0, 10, size=n_obs)

        # Use the same path for output to test in-place modification
        result_path = add_obs_column_to_h5ad(
            first_copy,
            first_copy,  # Same path - should modify in place
            column_name="second_column",
            column_data=custom_data,
        )

        # Verify both columns exist
        adata = ad.read_h5ad(result_path)
        assert "sample_index" in adata.obs.columns
        assert "second_column" in adata.obs.columns

        # Verify the data for both columns
        np.testing.assert_array_equal(adata.obs["sample_index"], np.arange(n_obs))
        np.testing.assert_array_equal(adata.obs["second_column"], custom_data)


@pytest.mark.skipif(
    os.environ.get("SKIP_LARGE_TESTS") == "1",
    reason="Skip large tests by setting SKIP_LARGE_TESTS=1",
)
def test_large_file_stress_test():
    """
    Create and modify a larger h5ad file to test performance and memory use with larger datasets.
    Skip this test with SKIP_LARGE_TESTS=1 environment variable.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a larger AnnData object (1000 cells x 5000 genes)
        n_obs = 1000
        n_vars = 5000

        # Use sparse matrix to save memory
        from scipy import sparse

        X = sparse.random(n_obs, n_vars, density=0.1, format="csr", random_state=42)

        # Create AnnData with sparse matrix
        adata = ad.AnnData(X=X)

        # Add some basic metadata
        adata.obs["batch"] = np.random.choice(
            ["batch1", "batch2", "batch3"], size=n_obs
        )
        adata.var["gene_type"] = np.random.choice(
            ["protein_coding", "lncRNA"], size=n_vars
        )

        # Save to temporary file
        temp_path = Path(tmpdir) / "large_test.h5ad"
        adata.write_h5ad(temp_path, compression="gzip")

        # Free memory
        del adata, X
        gc.collect()

        # Measure time to add column
        import time

        start_time = time.time()

        # Add sample index
        out_path = Path(tmpdir) / "large_out.h5ad"
        result_path = add_sample_index_to_h5ad(temp_path, out_path)

        end_time = time.time()
        elapsed = end_time - start_time

        # Log the performance
        print(f"\nLarge file modification took {elapsed:.2f} seconds")

        # Verify the file was modified correctly
        adata = ad.read_h5ad(result_path)
        assert "sample_index" in adata.obs.columns
        assert adata.n_obs == n_obs
        assert adata.n_vars == n_vars

        # Add another column in-place to test multiple modifications
        sample_groups = np.random.choice(["group_A", "group_B", "group_C"], size=n_obs)
        add_obs_column_to_h5ad(
            result_path,
            result_path,  # In-place modification
            column_name="sample_group",
            column_data=sample_groups,
            is_categorical=True,
        )

        # Verify both columns
        adata = ad.read_h5ad(result_path)
        assert "sample_index" in adata.obs.columns
        assert "sample_group" in adata.obs.columns
