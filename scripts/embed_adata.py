#!/usr/bin/env python
"""
Apply initial embeddings to preprocessed AnnData files.

This script:
1. Reads a preprocessed AnnData file
2. Applies the selected initial embedder
3. Saves the result in a new directory structure

References
----------
- Hydra: https://hydra.cc
- anndata: https://anndata.readthedocs.io
"""

import sys
from pathlib import Path
import hydra
from omegaconf import DictConfig
from dotenv import load_dotenv
from adata_hf_datasets.utils import setup_logging
from adata_hf_datasets.initial_embedder import InitialEmbedder
from adata_hf_datasets.sys_monitor import SystemMonitor
from hydra.core.hydra_config import HydraConfig


logger = setup_logging()


@hydra.main(version_base=None, config_path="../conf", config_name="embed_adata")
def main(cfg: DictConfig):
    """
    Main function for applying initial embeddings using Hydra config.

    Parameters
    ----------
    cfg : DictConfig
        Configuration object with fields:
        - input_file (str): Path to preprocessed .h5ad file
        - method (str): Initial embedder method to use
        - batch_key (str): Key in adata.obs for batch information
        - chunk_size (int): Number of cells to process at once
        - embedding_dim (int): Dimension of the embedding
    """
    load_dotenv(override=True)
    hydra_run_dir = HydraConfig.get().run.dir

    # Setup monitoring
    monitor = SystemMonitor(logger=logger)
    monitor.start()
    try:
        for input_file in cfg.input_files:
            input_file = Path(input_file)
            logger.info(f"Processing input file: {input_file}")
            if not input_file.is_file():
                raise FileNotFoundError(f"Input file not found: {input_file}")

            # Create output directory structure
            # Replace 'processed' with 'processed_with_emb' in the path
            output_dir = Path(
                str(input_file.parent).replace("processed", "processed_with_emb")
            )
            output_dir.mkdir(parents=True, exist_ok=True)

            # Create output filename with method name
            output_file = output_dir / f"{input_file.stem}_{cfg.method}.h5ad"

            if output_file.exists() and not cfg.overwrite:
                logger.info(
                    f"Output file already exists: {output_file}. Set overwrite=True to overwrite."
                )
                return

            # Initialize embedder
            embedding_dim = cfg.embedding_dim_map[cfg.method]
            embedder = InitialEmbedder(method=cfg.method, embedding_dim=embedding_dim)

            # Log event before preparation
            monitor.log_event(f"Starting preparation for {cfg.method}")

            # Prepare embedder
            embedder.prepare(adata_path=str(input_file), batch_key=cfg.batch_key)

            # Log event before embedding
            monitor.log_event(f"Starting embedding with {cfg.method}")

            # Apply embedding
            _ = embedder.embed(
                adata_path=str(input_file),
                output_path=str(output_file),
                chunk_size=cfg.chunk_size,
                batch_key=cfg.batch_key,
                batch_size=cfg.batch_size,
            )

            monitor.log_event(f"Finished embedding with {cfg.method}")
            logger.info(f"Saved embedded data to {output_file}")

    except Exception as e:
        logger.exception("An error occurred during embedding")
        raise e

    finally:
        # Stop monitoring and save results
        monitor.stop()
        monitor.print_summary()
        monitor.save(hydra_run_dir)
        monitor.plot_metrics(hydra_run_dir)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("An error occurred during embedding.")
        sys.exit(1)
# End of script
