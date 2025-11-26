# Workflow Orchestration

This directory contains scripts for orchestrating the complete single-cell data processing pipeline. The workflow orchestrator manages job submission, dependency tracking, logging, and error handling across multiple pipeline steps.

## Overview

The workflow orchestrator provides a unified interface for running all pipeline steps (download, preprocessing, embedding, dataset creation) with automatic:

- Job submission and dependency management
- Configuration resolution and propagation
- Centralized logging and error tracking
- Resource allocation (CPU vs GPU)
- Cancellation and cleanup

**For basic usage instructions, see the [Main README](../../README.md)**. This document provides technical details about how the orchestration works internally.

---

## Table of Contents

- [Architecture](#architecture)
- [Submission Process](#submission-process)
- [Configuration Resolution](#configuration-resolution)
- [Master Job Coordination](#master-job-coordination)
- [Logging and Error Handling](#logging-and-error-handling)
- [Job Cancellation](#job-cancellation)
- [Execution Modes](#execution-modes)
- [Troubleshooting](#troubleshooting)

---

## Architecture

### Components

The workflow system consists of several key components:

```
User Interface:
├── submit_workflow.py          # SLURM cluster submission
├── submit_workflow_local.py    # Local machine submission
└── run_workflow_master.py      # Entry point (called by both)

Core Orchestrator:
└── src/adata_hf_datasets/workflow/
    ├── workflow_orchestrator.py     # Main orchestration logic
    ├── config_utils.py              # Configuration transformations
    └── WorkflowLogger class         # Centralized logging

Step Scripts:
├── scripts/download/
├── scripts/preprocessing/
├── scripts/embed/
└── scripts/dataset_creation/
```

### Workflow Flow

```
1. User runs submit_workflow.py or submit_workflow_local.py
                    ↓
2. Script validates configuration and submits master job
                    ↓
3. Master job (run_workflow_master.py) starts
                    ↓
4. Master loads configs, resolves paths, creates workflow directory
                    ↓
5. Master executes each enabled step sequentially:
   - Download (optional)
   - Preprocessing
   - Embedding Preparation (CPU) (with array jobs for chunks)
   - CPU Embedding (with array jobs for chunks)
   - GPU Embedding (with array jobs for chunks)
   - Dataset Creation
                    ↓
6. Master collects logs, generates summary, marks completion
```

---

## Submission Process

### SLURM Submission (`submit_workflow.py`)

When you run:

```bash
python scripts/workflow/submit_workflow.py --config-name dataset_example
```

**What happens:**

1. **Load Configurations:**
   - Loads `conf/workflow_orchestrator.yaml` for cluster settings
   - Loads `conf/dataset_example.yaml` for dataset parameters

2. **Validate Config Synchronization:**
   - Checks that the dataset config exists on the remote cluster
   - Compares local and remote config file hashes
   - Ensures configs are in sync (unless `--force` is used)

3. **Submit Master SLURM Job:**
   - Constructs `sbatch` command with appropriate environment variables
   - SSHs to CPU cluster: `ssh cpu_cluster "cd {project_dir} && sbatch ..."`
   - Passes `DATASET_CONFIG` and `PROJECT_DIR` as environment variables
   - Returns master job ID

4. **Job Starts:**
   - SLURM scheduler allocates resources on CPU cluster
   - Runs `scripts/workflow/run_workflow_master.slurm`
   - Which in turn calls `scripts/workflow/run_workflow_master.py`

**Key Environment Variables:**

- `DATASET_CONFIG`: Name of dataset config (e.g., `dataset_example`)
- `PROJECT_DIR`: Project directory on cluster
- `SLURM_JOB_ID`: Assigned by SLURM, used as workflow identifier
- `EXECUTION_MODE`: Set to `slurm` for cluster execution

### Local Submission (`submit_workflow_local.py`)

When you run:

```bash
python scripts/workflow/submit_workflow_local.py --config-name dataset_example
```

**What happens:**

1. **Load Configuration:**
   - Loads `conf/workflow_orchestrator.yaml`
   - Sets `execution_mode` to `local`

2. **Generate Run ID:**
   - Creates unique run ID: `local_{timestamp}`
   - Used for log directory naming

3. **Run Master Process:**
   - **Foreground mode** (`--foreground`): Runs in current terminal
   - **Background mode** (default): Uses `nohup` + `caffeinate` (macOS) to run detached

4. **Execute Steps:**
   - Master process calls `run_workflow_localhost()` function
   - Steps execute as local subprocesses (not SLURM jobs)
   - Each step runs synchronously

**Key Differences from SLURM:**

- No SSH required (runs locally)
- No SLURM job dependencies (sequential execution)
- No cluster resource allocation
- Simpler for development and testing

---

## Configuration Resolution

One of the key features of the orchestrator is **automatic configuration resolution** for each step.

### The Problem

Different steps need different information:

- Preprocessing needs: input file path, output directory
- Embedding needs: processed file paths, embedding dimensions
- Dataset creation needs: embedded file paths, HuggingFace settings

But we want users to specify dataset properties **once** in a single config file.

### The Solution: Configuration Transformations

The `apply_all_transformations()` function (in `config_utils.py`) automatically:

1. **Generates file paths** from dataset metadata:

   ```python
   # User specifies once:
   dataset.name = "cellxgene_10k"
   base_file_path = "/data/RNA"
   split_dataset = true

   # Automatically generates:
   preprocessing.input_file = "/data/RNA/raw/train/cellxgene_10k.h5ad"
   preprocessing.output_dir = "/data/RNA/processed/train/cellxgene_10k"
   embedding_cpu.input_files = [
       "/data/RNA/processed/train/cellxgene_10k/train/chunk_0.zarr",
       "/data/RNA/processed/train/cellxgene_10k/val/chunk_0.zarr"
   ]
   embedding_cpu.output_dir = "/data/RNA/processed_with_emb/train/cellxgene_10k"
   ```

2. **Propagates common keys** to all steps:

   ```python
   # User defines once at top level:
   batch_key = "dataset_title"
   annotation_key = "cell_type"

   # Automatically copied to:
   preprocessing.batch_key = "dataset_title"
   embedding_cpu.batch_key = "dataset_title"
   dataset_creation.batch_key = "dataset_title"
   ```

3. **Auto-generates consolidation categories:**

   ```python
   # Based on annotation_key and other_bio_labels
   preprocessing.consolidation_categories = ["cell_type", "tissue", "disease"]
   ```

4. **Resolves variable interpolations:**

   ```python
   # Hydra variable references like:
   preprocessing.split_dataset = ${split_dataset}

   # Are resolved to actual values:
   preprocessing.split_dataset = true
   ```

### When Resolution Happens

```
Master job starts
       ↓
run_workflow_master.py loads config
       ↓
apply_all_transformations(config) is called
       ↓
Resolved config passed to each step
       ↓
Step scripts receive fully-resolved config with all paths
```

Each step script also calls `apply_all_transformations()` to ensure consistency.

---

## Master Job Coordination

The master job is the central coordinator that manages all pipeline steps.

### Master Job Responsibilities

1. **Configuration Management:**
   - Load and resolve dataset config
   - Set up environment variables (`BASE_FILE_PATH`, `WORKFLOW_DIR`)
   - Validate enabled/disabled steps

2. **Directory Structure:**
   - Create workflow directory: `{output_dir}/{date}/workflow_{job_id}/`
   - Create step subdirectories: `preprocessing/`, `embedding/`, etc.
   - Copy dataset config for reproducibility

3. **Step Execution:**
   - Check if each step is enabled in config
   - Execute or skip based on flags
   - Pass appropriate parameters and environment variables

4. **Logging:**
   - Create centralized log files
   - Redirect step outputs to appropriate directories
   - Generate workflow summary at completion

### Step Execution Patterns

#### SLURM Mode

For SLURM execution, the master job:

**Single-File Steps** (preprocessing, dataset creation):

```python
# Submit SLURM job
job_id = submit_slurm_job(
    script="scripts/preprocessing/run_preprocess.slurm",
    env_vars={"DATASET_CONFIG": dataset_config_name},
    dependency=None  # or previous_job_id
)

# Wait for completion (polling)
wait_for_job(job_id, poll_interval=3600)
```

**Array Job Steps** (embedding):

```python
# Discover input chunks
input_files = glob(f"{processed_dir}/train/chunk_*.zarr")

# Submit array job (one task per chunk)
array_job_id = submit_slurm_array_job(
    script="scripts/embed/embed_array.slurm",
    array_size=len(input_files),
    dependency=prep_job_id
)

# Wait for all array tasks
wait_for_job(array_job_id, poll_interval=3600)
```

#### Local Mode

For local execution, the master process:

**Single-File Steps:**

```python
# Run as subprocess
subprocess.run(
    ["python", "scripts/preprocessing/preprocess.py", "--config-name", dataset_config],
    cwd=project_root,
    stdout=log_file,
    stderr=err_file
)
# Blocks until complete
```

**Array Job Steps:**

```python
# Use ThreadPoolExecutor for parallel processing
with ThreadPoolExecutor(max_workers=local_max_workers) as executor:
    futures = []
    for input_file in input_files:
        future = executor.submit(
            process_chunk,
            input_file=input_file,
            config_name=dataset_config,
            ...
        )
        futures.append(future)

    # Wait for all to complete
    for future in futures:
        future.result()
```

### Job Dependencies (SLURM)

SLURM dependencies ensure correct execution order:

```
Download Job (if enabled)
    ↓ (afterok dependency)
Preprocessing Job
    ↓ (afterok)
Embedding Preparation Array Job
    ↓ (afterok)
┌───────────────────────────┐
│  CPU Embedding Array Job  │  (parallel, both wait for prep)
│  GPU Embedding Array Job  │
└───────────────────────────┘
    ↓ (afterok, waits for both)
Dataset Creation Job(s)
```

**Dependency types:**

- `afterok`: Next job runs only if previous succeeded (exit code 0)
- `afterany`: Next job runs regardless of previous exit code
- The orchestrator uses `afterok` to stop on failures

---

## Logging and Error Handling

### Log Directory Structure

```
{output_directory}/{date}/workflow_{job_id}/
├── logs/
│   ├── workflow_master.out              # Master job stdout
│   ├── workflow_master.err              # Master job stderr
│   ├── workflow_summary.log             # High-level summary
│   └── errors_consolidated.log          # All ERROR-level messages
│
├── preprocessing/
│   └── job_{slurm_job_id}/
│       ├── preprocessing.out
│       ├── preprocessing.err
│       └── .hydra/config.yaml           # Resolved config used
│
├── embedding_prepare/
│   └── job_{slurm_job_id}/
│       ├── master.out
│       ├── master.err
│       └── array_{array_job_id}/
│           ├── task_0.out
│           ├── task_0.err
│           ├── task_1.out
│           └── task_1.err
│
├── embedding/
│   ├── job_{cpu_job_id}/
│   │   ├── cpu_master.out
│   │   └── array_{array_job_id}/
│   │       └── task_*.out/err
│   └── job_{gpu_job_id}/
│       ├── gpu_master.out
│       └── array_{array_job_id}/
│           └── task_*.out/err
│
├── dataset_creation/
│   └── job_{slurm_job_id}/
│       ├── create_ds_0.out              # One per cs_length/caption_key combo
│       └── create_ds_0.err
│
└── config/
    └── dataset_config.yaml              # Copy of dataset config
```

### Consolidated Error Logging

The orchestrator attempts to consolidate all ERROR-level log messages:

**How it works:**

1. Each step script adds a logging handler for `errors_consolidated.log`
2. All `logger.error()` calls are written to this file
3. Master job can check this file for failures

**Example:**

```python
# In each step script
error_log_path = os.path.join(WORKFLOW_DIR, "logs", "errors_consolidated.log")
error_handler = logging.FileHandler(error_log_path, mode="a")
error_handler.setLevel(logging.ERROR)
logging.getLogger().addHandler(error_handler)
```

### Error Handling Limitations

⚠️ **Known Issues:**

**Problem 1: Jobs Don't Always Stop on Errors**

Sometimes a job reports an error but doesn't exit with a non-zero code:

- Python exceptions are caught and logged but script continues
- SLURM sees exit code 0 (success) and continues pipeline
- Dependent jobs run even though previous step failed

**Workaround:**

- Check `errors_consolidated.log` manually
- Look for ERROR messages in step-specific `.err` files
- Monitor job outputs during execution

**Problem 2: Silent Failures**

Some failures may not be logged to the consolidated log:

- Segmentation faults (process crashes)
- Out-of-memory kills (SLURM sends SIGKILL)
- Network errors during SSH/SLURM communication

**Workaround:**

- Check SLURM job status: `sacct -j {job_id}`
- Look for `FAILED`, `CANCELLED`, `OUT_OF_MEMORY` states
- Check step-specific `.err` files

**Problem 3: Resource Exhaustion**

Jobs may fail due to resource limits:

- Memory exceeded → SLURM kills job
- Time limit exceeded → Job cancelled
- Disk space full → Write failures

**Workaround:**

- Check `sacct -j {job_id} --format=JobID,State,MaxRSS,Elapsed`
- Increase memory allocation in step SLURM scripts
- Increase time limits for large datasets

### Debugging Workflow Issues

**Step 1: Check workflow summary**

```bash
cat {output_dir}/{date}/workflow_{job_id}/logs/workflow_summary.log
```

**Step 2: Check consolidated errors**

```bash
cat {output_dir}/{date}/workflow_{job_id}/logs/errors_consolidated.log
```

**Step 3: Check master job logs**

```bash
cat {output_dir}/{date}/workflow_{job_id}/logs/workflow_master.err
```

**Step 4: Check individual step logs**

```bash
# Find the failing step
ls {output_dir}/{date}/workflow_{job_id}/

# Check its error log
cat {output_dir}/{date}/workflow_{job_id}/{step_name}/job_*/step.err
```

**Step 5: Check SLURM job status (SLURM mode only)**

```bash
# On the cluster
sacct -j {job_id} --format=JobID,JobName,State,ExitCode,MaxRSS,Elapsed

# Look for non-zero ExitCode or FAILED State
```

---

## Job Cancellation

### Cancelling the Master Job

The master job controls all subjobs. **Cancelling the master job cancels everything.**

#### SLURM Mode

```bash
# Find the master job ID (from submission output or squeue)
ssh cpu_cluster "squeue -u username"

# Cancel the master job
ssh cpu_cluster "scancel {master_job_id}"
```

**What happens:**

1. SLURM sends SIGTERM to master job
2. Master job's signal handler catches it
3. Master job calls `cancel_all_jobs()`:
   - Cancels all subjobs it submitted
   - Cancels preprocessing job (if running)
   - Cancels embedding array jobs (if running)
   - Cancels dataset creation jobs (if running)
4. Master job exits

**Automatic cleanup:**

- All SLURM jobs with dependencies on the master are automatically cancelled by SLURM
- This includes queued jobs that haven't started yet

#### Local Mode

```bash
# Find the process (if running in background)
ps aux | grep run_workflow_master.py

# Kill the process
kill {pid}

# Or use the PID file (if you noted the output directory)
kill $(cat {output_dir}/{date}/workflow_{timestamp}/master.pid)
```

**What happens:**

1. Python process receives SIGTERM
2. Signal handler catches it (if running in foreground)
3. All subprocess are terminated
4. Process exits

**Note:** Background processes (using `nohup`) may not respond to Ctrl+C. Use `kill` command instead.

### Cancelling Individual Steps

⚠️ **Not recommended:** Cancelling individual subjobs breaks the workflow dependency chain.

**What happens if you cancel a subjob:**

- Master job may not detect the cancellation
- Dependent jobs may fail due to missing input files
- Master job may hang waiting for a job that will never complete

**If you must cancel a specific step:**

1. Cancel the master job first
2. Or let the step fail naturally and the master will handle it

---

## Execution Modes

### SLURM Mode

**Best for:**

- Large datasets (> 100K cells)
- Production workflows
- GPU-intensive steps
- Parallel processing across multiple nodes

**Requirements:**

- SLURM cluster access
- SSH key authentication
- Shared filesystem between CPU and GPU nodes
- Appropriate partition access

**Configuration:**

```yaml
workflow:
  execution_mode: "slurm"
  cpu_login:
    host: "cpu_cluster"
    user: "username"
  gpu_login:
    host: "gpu_cluster"
    user: "username"
  slurm_base_file_path: "/scratch/global/username/data/RNA"
```

### Local Mode

**Best for:**

- Small datasets (< 10K cells)
- Development and testing
- Debugging workflows
- Single-machine setups

**Requirements:**

- Sufficient local resources (RAM, CPU)
- Python environment with all dependencies
- Optional: CUDA-capable GPU (for GPU embeddings)

**Configuration:**

```yaml
workflow:
  execution_mode: "local"
  local_base_file_path: "/Users/username/data/RNA"
  local_max_workers: 2
  local_enable_gpu: false
```

**Limitations:**

- No true parallelization (uses threading, not multiprocessing)
- Limited by local machine resources
- GPU embedding disabled by default (to prevent OOM on laptops)

---

## Troubleshooting

### Common Issues

#### Issue: "SSH connection timed out"

**Cause:** Cannot connect to cluster via SSH

**Solution:**

```bash
# Test SSH connection
ssh cpu_cluster "hostname"

# Check SSH config
cat ~/.ssh/config

# Ensure SSH agent has keys
ssh-add -l
ssh-add ~/.ssh/id_ed25519
```

#### Issue: "Config file not synchronized"

**Cause:** Local and remote configs differ

**Solution:**

```bash
# Option 1: Use --force to skip check
python scripts/workflow/submit_workflow.py --config-name dataset_example --force

# Option 2: Sync configs manually
ssh cpu_cluster "cd {project_dir} && git pull"

# Option 3: Copy config directly
scp conf/dataset_example.yaml cpu_cluster:{project_dir}/conf/
```

#### Issue: "Master job stuck waiting"

**Cause:** A subjob failed but master didn't detect it

**Solution:**

```bash
# Check subjob status
ssh cpu_cluster "sacct -j {master_job_id}.batch --format=JobID,State"

# Look for FAILED or CANCELLED
# Cancel master job if needed
ssh cpu_cluster "scancel {master_job_id}"
```

#### Issue: "No such file or directory" during workflow

**Cause:** Path configuration mismatch

**Solution:**

- Check `base_file_path` in `workflow_orchestrator.yaml`
- Ensure it's accessible from both CPU and GPU clusters
- Verify previous steps completed successfully
- Check step-specific output directories exist

#### Issue: "Permission denied" errors

**Cause:** Insufficient permissions on shared filesystem

**Solution:**

```bash
# Check permissions
ssh cpu_cluster "ls -la {base_file_path}"

# Create directory if needed
ssh cpu_cluster "mkdir -p {base_file_path}/raw/train"

# Check if writable
ssh cpu_cluster "touch {base_file_path}/test && rm {base_file_path}/test"
```

#### Issue: "Import error" in step scripts

**Cause:** Virtual environment not activated or missing dependencies

**Solution:**

```bash
# On cluster, verify venv
ssh cpu_cluster "cd {project_dir} && source .venv/bin/activate && python -c 'import adata_hf_datasets'"

# Reinstall if needed
ssh cpu_cluster "cd {project_dir} && uv sync --all-extras"
```

### Getting Help

For workflow-specific issues:

1. **Check master job logs:**
   - `{output_dir}/{date}/workflow_{job_id}/logs/workflow_master.err`

2. **Check consolidated errors:**
   - `{output_dir}/{date}/workflow_{job_id}/logs/errors_consolidated.log`

3. **Check step-specific logs:**
   - Navigate to the failing step's directory and check `.err` files

4. **Check SLURM job status:**
   - `sacct -j {job_id} --format=JobID,State,ExitCode,MaxRSS`

5. **Review configuration:**
   - `{output_dir}/{date}/workflow_{job_id}/config/dataset_config.yaml`

For step-specific issues, see the individual step README files:

- [Download README](../download/README.md)
- [Preprocessing README](../preprocessing/README.md)
- [Embedding README](../embed/README.md)
- [Dataset Creation README](../dataset_creation/README.md)

---

## Summary

The workflow orchestrator provides a robust system for managing complex multi-step pipelines:

✅ **Automatic configuration resolution** from dataset metadata
✅ **Centralized logging** with consolidated error tracking
✅ **Job dependency management** via SLURM or sequential execution
✅ **Cancellation support** with automatic cleanup
✅ **Multiple execution modes** for different use cases

⚠️ **Known limitations:**

- Error detection is not perfect - manual log checking may be needed
- Some failures may not stop the pipeline automatically
- Resource exhaustion can cause silent failures

For most use cases, follow the [Main README](../../README.md) for basic usage. Refer to this document for troubleshooting and understanding the internal workings.
