"""
Configuration utilities for dataset-centric workflow management.

This module provides utilities to:
1. Generate paths automatically from dataset metadata
2. Extract workflow-specific configs from dataset configs
3. Validate configs and ensure required fields are present
4. Auto-generate consolidation categories and categories of interest from common keys
5. Apply transformations to resolved config objects from Hydra
"""

from pathlib import Path
from typing import Dict, Any, List
from omegaconf import DictConfig, OmegaConf
import logging
import subprocess

logger = logging.getLogger(__name__)


def generate_paths_from_config(cfg: DictConfig) -> Dict[str, str]:
    """
    Generate all required paths from dataset metadata in a resolved config.

    Parameters
    ----------
    cfg : DictConfig
        The complete dataset configuration (resolved by Hydra)

    Returns
    -------
    Dict[str, str]
        Dictionary mapping path keys to generated paths
    """
    dataset_name = cfg.dataset.name
    full_name = cfg.dataset.full_name
    base_file_path = cfg.get("base_file_path", "data/RNA/raw")

    # Determine if this is a training or test dataset
    is_training = cfg.preprocessing.split_dataset

    # Get the output format from preprocessing config
    output_format = cfg.preprocessing.get("output_format", "zarr")

    # Base directories for processed and embedded data
    if is_training:
        raw_base = "data/RNA/raw/train"
        processed_base = "data/RNA/processed/train"
        embed_base = "data/RNA/processed_with_emb/train"
    else:
        raw_base = "data/RNA/raw/test"
        processed_base = "data/RNA/processed/test"
        embed_base = "data/RNA/processed_with_emb/test"

    # Generate paths
    paths = {
        # Download paths - use full_name if specified, otherwise auto-generate
        "download.full_file_path": f"{base_file_path}/{full_name}.h5ad"
        if full_name
        else f"{raw_base}/{dataset_name}_full.h5ad",
        "download.output_path": f"{base_file_path}/{dataset_name}.h5ad",
        # Dataset file path - auto-generated from base_file_path and name
        "dataset.file_path": f"{raw_base}/{dataset_name}.h5ad",
        # Preprocessing paths
        "preprocessing.input_file": f"{raw_base}/{dataset_name}.h5ad",
        "preprocessing.output_dir": f"{processed_base}/{dataset_name}",
        # Embedding paths
        "embedding.output_dir": f"{embed_base}/{dataset_name}",
        # Dataset creation paths
        "dataset_creation.data_dir": f"{embed_base}/{dataset_name}",
    }

    # Generate input files for embedding based on preprocessing output
    if is_training:
        # Training datasets have train/val splits
        paths["embedding.input_files"] = [
            f"{processed_base}/{dataset_name}/train/chunk_0.{output_format}",
            f"{processed_base}/{dataset_name}/val/chunk_0.{output_format}",
        ]
    else:
        # Test datasets have a single "all" split
        paths["embedding.input_files"] = [
            f"{processed_base}/{dataset_name}/all/chunk_0.{output_format}"
        ]

    return paths


def apply_path_transformations(cfg: DictConfig) -> DictConfig:
    """
    Apply path transformations to a resolved config object.

    This function respects command-line overrides. If a path has been explicitly
    set via command line (e.g., ++input_files or ++embedding.input_files), it will
    not be overwritten by auto-generated paths.

    Parameters
    ----------
    cfg : DictConfig
        The complete dataset configuration (resolved by Hydra)

    Returns
    -------
    DictConfig
        Updated configuration with generated paths
    """
    paths = generate_paths_from_config(cfg)

    # Create a copy of the config
    updated_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))

    # Handle top-level input_files override (for backward compatibility with SLURM scripts)
    if hasattr(cfg, "input_files") and cfg.input_files is not None:
        logger.debug(f"Found top-level input_files override: {cfg.input_files}")
        updated_cfg.embedding.input_files = cfg.input_files

    # Update paths in the config, but respect command-line overrides
    for path_key, path_value in paths.items():
        keys = path_key.split(".")
        current = updated_cfg
        original_current = cfg

        # Navigate to the parent of the target key in both configs
        for key in keys[:-1]:
            current = current[key]
            original_current = original_current[key]

        target_key = keys[-1]

        # Check if this field was explicitly set in the original config
        # This indicates a command-line override
        if (
            hasattr(original_current, target_key)
            and original_current[target_key] is not None
        ):
            logger.debug(
                f"Skipping auto-generation of {path_key} - already set to: {original_current[target_key]}"
            )
            continue

        # Apply the auto-generated path
        current[target_key] = path_value
        logger.debug(f"Applied auto-generated path: {path_key} = {path_value}")

    return updated_cfg


def generate_stratify_keys(cfg: DictConfig) -> List[str]:
    """
    Generate stratify keys from common keys.

    Parameters
    ----------
    cfg : DictConfig
        The complete dataset configuration

    Returns
    -------
    List[str]
        List of stratify keys: [batch_key, annotation_key] (if both are present)
    """
    keys = []

    # Add batch_key if present
    if cfg.get("batch_key") is not None:
        keys.append(cfg.batch_key)

    # Add annotation_key if present
    if cfg.get("annotation_key") is not None:
        keys.append(cfg.annotation_key)

    return keys


def apply_common_key_transformations(cfg: DictConfig) -> DictConfig:
    """
    Apply common key transformations to a resolved config object.

    Parameters
    ----------
    cfg : DictConfig
        The complete dataset configuration (resolved by Hydra)

    Returns
    -------
    DictConfig
        Updated configuration with auto-generated values
    """
    # Create a copy of the config
    updated_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))

    # Generate consolidation categories
    consolidation_categories = generate_consolidation_categories(updated_cfg)
    if consolidation_categories:
        updated_cfg.preprocessing.consolidation_categories = consolidation_categories

    # Generate categories of interest
    categories_of_interest = generate_categories_of_interest(updated_cfg)
    if categories_of_interest:
        updated_cfg.preprocessing.categories_of_interest = categories_of_interest

    # Generate stratify keys for download
    stratify_keys = generate_stratify_keys(updated_cfg)
    if stratify_keys:
        updated_cfg.download.stratify_keys = stratify_keys

    # Update batch_key in preprocessing config (for gene selection)
    if updated_cfg.get("batch_key") is not None:
        updated_cfg.preprocessing.batch_key = updated_cfg.batch_key

    # Update batch_key in embedding config
    if updated_cfg.get("batch_key") is not None:
        updated_cfg.embedding.batch_key = updated_cfg.batch_key

    # Update annotation_key in dataset_creation config
    if updated_cfg.get("annotation_key") is not None:
        updated_cfg.dataset_creation.annotation_key = updated_cfg.annotation_key

    # Update caption_key in dataset_creation config
    if updated_cfg.get("caption_key") is not None:
        updated_cfg.dataset_creation.caption_key = updated_cfg.caption_key

    # Update instrument_key in preprocessing config
    if updated_cfg.get("instrument_key") is not None:
        updated_cfg.preprocessing.instrument_key = updated_cfg.instrument_key

    return updated_cfg


def apply_all_transformations(cfg: DictConfig) -> DictConfig:
    """
    Apply all transformations to a resolved config object.

    Parameters
    ----------
    cfg : DictConfig
        The complete dataset configuration (resolved by Hydra)

    Returns
    -------
    DictConfig
        Updated configuration with all transformations applied
    """
    # Apply common key transformations first
    cfg = apply_common_key_transformations(cfg)

    # Then apply path transformations
    cfg = apply_path_transformations(cfg)

    return cfg


def extract_workflow_config(cfg: DictConfig, workflow: str) -> DictConfig:
    """
    Extract workflow-specific configuration from dataset config.

    Parameters
    ----------
    cfg : DictConfig
        The complete dataset configuration
    workflow : str
        The workflow name: "preprocessing", "embedding", or "dataset_creation"

    Returns
    -------
    DictConfig
        Configuration specific to the workflow
    """
    if workflow not in ["preprocessing", "embedding", "dataset_creation"]:
        raise ValueError(f"Unknown workflow: {workflow}")

    # Extract the workflow-specific config
    workflow_cfg = cfg[workflow]

    # Add dataset metadata if needed
    if workflow in ["preprocessing", "embedding"]:
        workflow_cfg.dataset = cfg.dataset

    return workflow_cfg


def validate_config(cfg: DictConfig) -> bool:
    """
    Validate that the configuration has all required fields.

    Parameters
    ----------
    cfg : DictConfig
        The dataset configuration to validate

    Returns
    -------
    bool
        True if valid, raises ValueError if invalid
    """
    required_fields = {
        "dataset": ["name", "file_path"],
        "preprocessing": ["input_file", "output_dir"],
        "embedding": ["input_files", "output_dir", "methods"],
        "dataset_creation": ["data_dir", "sentence_keys", "required_obsm_keys"],
    }

    for section, fields in required_fields.items():
        if section not in cfg:
            raise ValueError(f"Missing required section: {section}")

        for field in fields:
            if field not in cfg[section]:
                raise ValueError(f"Missing required field: {section}.{field}")

    return True


def get_dataset_info(cfg: DictConfig) -> Dict[str, Any]:
    """
    Extract basic dataset information from config.

    Parameters
    ----------
    cfg : DictConfig
        The dataset configuration

    Returns
    -------
    Dict[str, Any]
        Dictionary with dataset information
    """
    return {
        "name": cfg.dataset.name,
        "description": cfg.dataset.description,
        "download_url": cfg.dataset.download_url,
        "file_path": cfg.dataset.file_path,
        "is_training": cfg.preprocessing.split_dataset,
        "embedding_methods": cfg.embedding.methods,
        "dataset_format": cfg.dataset_creation.dataset_format,
        "batch_key": cfg.get("batch_key"),
        "annotation_key": cfg.get("annotation_key"),
        "caption_key": cfg.get("caption_key"),
        "instrument_key": cfg.get("instrument_key"),
        "other_bio_labels": cfg.get("other_bio_labels", []),
    }


def generate_consolidation_categories(cfg: DictConfig) -> List[str]:
    """
    Generate consolidation categories from common keys.

    Parameters
    ----------
    cfg : DictConfig
        The complete dataset configuration

    Returns
    -------
    List[str]
        List of consolidation categories: [batch_key, annotation_key]
    """
    categories = []

    # Add batch_key if present
    if cfg.get("batch_key") is not None:
        categories.append(cfg.batch_key)

    # Add annotation_key if present
    if cfg.get("annotation_key") is not None:
        categories.append(cfg.annotation_key)

    return categories


def generate_categories_of_interest(cfg: DictConfig) -> List[str]:
    """
    Generate categories of interest from common keys.

    Parameters
    ----------
    cfg : DictConfig
        The complete dataset configuration

    Returns
    -------
    List[str]
        List of categories of interest: [batch_key, annotation_key, instrument_key, other_bio_labels]
    """
    categories = []

    # Add batch_key if present
    if cfg.get("batch_key") is not None:
        categories.append(cfg.batch_key)

    # Add annotation_key if present
    if cfg.get("annotation_key") is not None:
        categories.append(cfg.annotation_key)

    # Add instrument_key if present
    if cfg.get("instrument_key") is not None:
        categories.append(cfg.instrument_key)

    # Add other_bio_labels if present
    if cfg.get("other_bio_labels") is not None:
        categories.extend(cfg.other_bio_labels)

    return categories


# Legacy functions for backward compatibility (deprecated)
def generate_paths(cfg: DictConfig) -> Dict[str, str]:
    """Deprecated: Use generate_paths_from_config instead."""
    logger.warning(
        "generate_paths is deprecated, use generate_paths_from_config instead"
    )
    return generate_paths_from_config(cfg)


def update_config_with_paths(cfg: DictConfig) -> DictConfig:
    """Deprecated: Use apply_path_transformations instead."""
    logger.warning(
        "update_config_with_paths is deprecated, use apply_path_transformations instead"
    )
    return apply_path_transformations(cfg)


def update_config_with_common_keys(cfg: DictConfig) -> DictConfig:
    """Deprecated: Use apply_common_key_transformations instead."""
    logger.warning(
        "update_config_with_common_keys is deprecated, use apply_common_key_transformations instead"
    )
    return apply_common_key_transformations(cfg)


def create_workflow_configs(cfg_path: str) -> Dict[str, DictConfig]:
    """Deprecated: Use Hydra decorator and apply_all_transformations instead."""
    logger.warning(
        "create_workflow_configs is deprecated, use Hydra decorator and apply_all_transformations instead"
    )

    # Load the dataset config
    cfg = OmegaConf.load(cfg_path)

    # Apply transformations
    cfg = apply_all_transformations(cfg)

    # Validate the config
    validate_config(cfg)

    # Extract workflow-specific configs
    workflows = ["preprocessing", "embedding", "dataset_creation"]
    workflow_configs = {}

    for workflow in workflows:
        workflow_configs[workflow] = extract_workflow_config(cfg, workflow)

    return workflow_configs


def save_workflow_configs(cfg_path: str, output_dir: str):
    """Deprecated: Use Hydra decorator and apply_all_transformations instead."""
    logger.warning(
        "save_workflow_configs is deprecated, use Hydra decorator and apply_all_transformations instead"
    )

    workflow_configs = create_workflow_configs(cfg_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get dataset name for file naming
    cfg = OmegaConf.load(cfg_path)
    dataset_name = cfg.dataset.name

    for workflow, config in workflow_configs.items():
        output_file = output_path / f"{workflow}_{dataset_name}.yaml"
        OmegaConf.save(config, output_file)
        logger.info(f"Saved {workflow} config to {output_file}")


def validate_remote_config_sync(
    config_name: str,
    remote_host: str,
    remote_project_dir: str = "/home/menger/git/adata_hf_datasets",
) -> bool:
    """
    Validate that the remote config file matches the local one.

    This function compares the local config file with the remote one to ensure
    they are in sync, preventing workflow runs with outdated configurations.

    Parameters
    ----------
    config_name : str
        Name of the config file (without .yaml extension)
    remote_host : str
        SSH hostname for the remote server
    remote_project_dir : str
        Path to the project directory on the remote server

    Returns
    -------
    bool
        True if configs match, False otherwise

    Raises
    ------
    RuntimeError
        If unable to read local or remote config files
    """
    # Local config path
    local_config_path = Path("conf") / f"{config_name}.yaml"
    if not local_config_path.exists():
        raise RuntimeError(f"Local config file not found: {local_config_path}")

    # Read local config content
    with open(local_config_path, "r") as f:
        local_content = f.read()

    # Read remote config content
    remote_config_path = f"{remote_project_dir}/conf/{config_name}.yaml"
    cmd = ["ssh", remote_host, f"cat {remote_config_path}"]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to read remote config: {result.stderr}")
        remote_content = result.stdout
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Timeout reading remote config from {remote_host}")

    # Compare contents
    if local_content.strip() == remote_content.strip():
        return True
    else:
        return False


def ensure_config_sync(
    config_name: str,
    remote_host: str,
    remote_project_dir: str = "/home/menger/git/adata_hf_datasets",
    force: bool = False,
) -> None:
    """
    Ensure that the remote config file matches the local one.

    This function validates config synchronization and provides helpful error messages
    if the configs don't match.

    Parameters
    ----------
    config_name : str
        Name of the config file (without .yaml extension)
    remote_host : str
        SSH hostname for the remote server
    remote_project_dir : str
        Path to the project directory on the remote server
    force : bool
        If True, skip the validation check

    Raises
    ------
    RuntimeError
        If configs don't match and force=False
    """
    if force:
        logger.warning("Skipping config synchronization check (force=True)")
        return

    logger.info(f"Validating config synchronization for {config_name}...")

    try:
        if validate_remote_config_sync(config_name, remote_host, remote_project_dir):
            logger.info(
                f"✓ Config {config_name} is synchronized between local and remote"
            )
        else:
            error_msg = f"""
Config synchronization failed for {config_name}!

The local config file differs from the remote one. This could cause the workflow
to run with outdated configuration.

To fix this:
1. Commit and push your local changes: git add conf/{config_name}.yaml && git commit -m "Update {config_name} config" && git push
2. Pull the changes on the remote server: ssh {remote_host} "cd {remote_project_dir} && git pull"
3. Or run with --force to skip this check (not recommended)

Local config: conf/{config_name}.yaml
Remote config: {remote_project_dir}/conf/{config_name}.yaml
"""
            raise RuntimeError(error_msg)
    except Exception as e:
        if "Failed to read remote config" in str(e):
            error_msg = f"""
Remote config file not found or inaccessible!

The remote config file could not be read from {remote_host}.
This might indicate:
1. The config file doesn't exist on the remote server
2. SSH connection issues
3. Incorrect remote project directory

Remote path: {remote_project_dir}/conf/{config_name}.yaml
SSH host: {remote_host}

To fix this:
1. Ensure your code is synced to the remote server
2. Check the remote project directory path
3. Verify SSH connectivity
"""
            raise RuntimeError(error_msg)
        else:
            raise
