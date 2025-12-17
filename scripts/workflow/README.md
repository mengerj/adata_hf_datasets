# Workflow Orchestration

This directory contains scripts for orchestrating the complete single-cell data processing pipeline. The workflow orchestrator manages job submission, dependency tracking, logging, and error handling across multiple pipeline steps.

## Overview

The workflow orchestrator provides a **unified interface** for running all pipeline steps (download, preprocessing, embedding, dataset creation) with:

- **Per-step execution locations** - Each step can run on local, CPU cluster, or GPU cluster
- **Automatic data transfer** - Data is automatically moved between locations as needed
- **Centralized logging** - All logs collected in a single workflow directory
- **Graceful termination** - Kill signals propagate to all child processes and remote jobs
- **Resource allocation** - Appropriate CPU/GPU resources per step

**For basic usage instructions, see the [Main README](../../README.md)**. This document provides technical details about how the orchestration works internally.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Configuration](#configuration)
- [Execution Locations](#execution-locations)
- [Data Transfer](#data-transfer)
- [Logging and Monitoring](#logging-and-monitoring)
- [Job Cancellation](#job-cancellation)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

### Basic Usage

```bash
# Submit workflow (runs in background)
python scripts/workflow/submit_workflow.py --config human_pancreas

# Run in foreground (see output directly)
python scripts/workflow/submit_workflow.py --config human_pancreas --foreground

# Skip config sync validation
python scripts/workflow/submit_workflow.py --config human_pancreas --force
```

### Output

```
================================================================================
WORKFLOW SUBMISSION
================================================================================
Dataset config: human_pancreas
Force mode: False
Foreground: False

Loading workflow configuration...
✓ Workflow configuration loaded

Configured locations:
  local: local
  cpu: menger@imbi13
  gpu: menger@imbi_gpu_H100

Execution plan:
----------------------------------------
  download                  → local
  preprocessing             → cpu
  embedding_preparation     (disabled)
  embedding_cpu             → cpu
  embedding_gpu             → gpu
  dataset_creation          → cpu
----------------------------------------

Submitting workflow for dataset: human_pancreas
Submitting workflow in background...

================================================================================
WORKFLOW RUNNING IN BACKGROUND
================================================================================
Process ID (PID): 12345
Workflow directory: /path/to/outputs/2025-12-17/workflow_20251217_110830
Logs: /path/to/outputs/2025-12-17/workflow_20251217_110830/logs/

To stop this workflow: kill 12345
Or use: kill $(cat /path/to/outputs/2025-12-17/workflow_20251217_110830/logs/workflow.pid)

Note: Killing the process will also terminate any running steps
      (including remote SLURM jobs if applicable)
================================================================================
```

---

## Architecture

### Unified Workflow Model

The workflow system uses a **unified orchestrator** that can execute steps on different locations:

```
┌─────────────────────────────────────────────────────────────────┐
│                      WorkflowRunner                              │
│                                                                  │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │   Local     │    │     CPU     │    │     GPU     │         │
│  │  Executor   │    │  Executor   │    │  Executor   │         │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘         │
│         │                  │                  │                 │
│         ▼                  ▼                  ▼                 │
│    subprocess         SSH + SLURM        SSH + SLURM           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │       DataTransfer            │
              │  (rsync + tar.gz compression) │
              └───────────────────────────────┘
```

### Components

```
User Interface:
└── scripts/workflow/
    └── submit_workflow.py      # Unified submission script

Core Orchestrator:
└── src/adata_hf_datasets/workflow/
    ├── workflow_runner.py      # Main WorkflowRunner class
    ├── executors.py            # LocalExecutor, RemoteExecutor
    ├── data_transfer.py        # Data transfer between locations
    ├── log_retrieval.py        # Log consolidation utilities
    ├── config_utils.py         # Configuration transformations
    └── workflow_orchestrator.py # Legacy support + helpers

Step Scripts:
├── scripts/download/
├── scripts/preprocessing/
├── scripts/embed/
└── scripts/dataset_creation/
```

### Workflow Flow

```
1. User runs submit_workflow.py --config dataset_name
                    ↓
2. Script shows execution plan (which steps run where)
                    ↓
3. Validates config sync with remote locations (if needed)
                    ↓
4. Launches WorkflowRunner (foreground or background)
                    ↓
5. For each enabled step:
   a. Determine execution location from step config
   b. Transfer data if location changed from previous step
   c. Execute step using appropriate executor
   d. Collect logs to central workflow directory
                    ↓
6. Generate workflow summary
```

---

## Configuration

### Dataset Configuration

Each step in your dataset config can specify where it should run:

```yaml
# conf/my_dataset.yaml
defaults:
  - dataset_default.yaml
  - _self_

dataset:
  name: "my_dataset"
  description: "My dataset description"

# Per-step execution locations
download:
  enabled: true
  execution_location: local # Download locally (has internet)

preprocessing:
  enabled: true
  execution_location: cpu # Preprocess on CPU cluster

embedding_preparation:
  enabled: true
  execution_location: cpu

embedding_cpu:
  enabled: true
  execution_location: cpu
  methods: ["pca", "scvi_fm"]

embedding_gpu:
  enabled: true
  execution_location: gpu # GPU embeddings on GPU cluster
  methods: ["geneformer"]

dataset_creation:
  enabled: true
  execution_location: cpu
```

**Execution location options:**

- `local` - Run on the local machine
- `cpu` - Run on the CPU cluster via SLURM
- `gpu` - Run on the GPU cluster via SLURM

### Workflow Configuration

Configure the locations in `conf/workflow_orchestrator.yaml`:

```yaml
workflow:
  default_execution_location: local # Fallback if step doesn't specify

  locations:
    local:
      base_file_path: "./data/RNA"
      project_directory: "."
      venv_path: ".venv"
      output_directory: "./outputs"
      max_workers: 2
      enable_gpu: true

    cpu:
      ssh_host: "cpu_cluster" # SSH hostname
      ssh_user: "username" # SSH username
      slurm_partition: "slurm" # SLURM partition
      node: null # Specific node (optional)
      base_file_path: "/scratch/data/RNA"
      project_directory: "/home/user/project"
      venv_path: ".venv"
      output_directory: "/home/user/project/outputs"

    gpu:
      ssh_host: "gpu_cluster"
      ssh_user: "username"
      slurm_partition: "gpu"
      node: null
      base_file_path: "/scratch/data/RNA"
      project_directory: "/home/user/project"
      venv_path: ".venv"
      output_directory: "/home/user/project/outputs"

  transfer:
    enabled: true
    compression: true # Use tar.gz for Zarr directories
    compression_level: 6
    verify_integrity: true
    rsync_options:
      - "--archive"
      - "--compress"
      - "--partial"
      - "--progress"
      - "--human-readable"

  poll_interval: 120 # Seconds between SLURM job status checks
  job_timeout: 0 # Max wait time (0 = unlimited)
```

---

## Execution Locations

### Local Execution

When a step runs locally:

1. `LocalExecutor` spawns a subprocess
2. Output is captured to step-specific log files
3. Process is tracked for cleanup on termination

```python
# Internal execution
subprocess.Popen(
    ["python", "scripts/preprocessing/preprocess.py", "--config-name", "my_dataset"],
    stdout=log_file,
    stderr=err_file,
    cwd=project_directory
)
```

### Remote Execution (CPU/GPU)

When a step runs on a cluster:

1. `RemoteExecutor` generates a SLURM script
2. Submits via SSH: `ssh cluster "sbatch script.slurm"`
3. Polls job status until completion
4. Retrieves logs via rsync

```bash
# Generated SLURM script
#!/bin/bash
#SBATCH --job-name=preprocessing_20251217_110830
#SBATCH --output=/tmp/workflow_logs/preprocessing_20251217_110830.out
#SBATCH --error=/tmp/workflow_logs/preprocessing_20251217_110830.err
#SBATCH --partition=slurm

cd /home/user/project
source .venv/bin/activate
python scripts/preprocessing/preprocess.py --config-name my_dataset
```

### Mixed Execution Example

A typical workflow might use different locations for different steps:

| Step                  | Location | Reason                               |
| --------------------- | -------- | ------------------------------------ |
| download              | local    | Local machine has internet access    |
| preprocessing         | cpu      | CPU-intensive, benefits from cluster |
| embedding_preparation | cpu      | Prepare tokenized data               |
| embedding_cpu         | cpu      | PCA, scVI don't need GPU             |
| embedding_gpu         | gpu      | Geneformer needs GPU                 |
| dataset_creation      | cpu      | Final assembly, push to HuggingFace  |

---

## Data Transfer

### Automatic Transfer

When consecutive steps run on different locations, data is automatically transferred:

```
Step 1: download (local)
    ↓
    Output: ./data/RNA/raw/train/my_dataset.h5ad
    ↓
[TRANSFER: local → cpu via rsync]
    ↓
Step 2: preprocessing (cpu)
    ↓
    Output: /scratch/data/RNA/processed/train/my_dataset/
    ↓
[NO TRANSFER: cpu → cpu]
    ↓
Step 3: embedding_cpu (cpu)
    ↓
    Output: /scratch/data/RNA/processed_with_emb/train/my_dataset/
    ↓
[TRANSFER: cpu → gpu via rsync]
    ↓
Step 4: embedding_gpu (gpu)
```

### Transfer Mechanism

1. **Path Translation**: Paths are mapped between location base directories
   - `./data/RNA/raw/train/file.h5ad` (local)
   - → `/scratch/data/RNA/raw/train/file.h5ad` (cpu)

2. **Compression**: Zarr directories (many small files) are compressed with tar.gz before transfer

3. **Rsync**: Actual transfer uses rsync with progress tracking

   ```bash
   rsync --archive --compress --partial --progress \
       source:/path/to/data target:/path/to/data
   ```

4. **Verification**: Optional integrity check after transfer

### Transfer Configuration

```yaml
transfer:
  enabled: true # Enable automatic data transfer
  compression: true # Compress Zarr directories
  compression_level: 6 # gzip compression level (1-9)
  verify_integrity: true # Verify after transfer
  remote_to_remote_via_local: false # Direct remote-to-remote if SSH agent forwarding works
  temp_dir: "/tmp/workflow_transfer"
  cleanup_temp: true
```

---

## Logging and Monitoring

### Directory Structure

All logs are centralized in a single workflow directory:

```
outputs/2025-12-17/workflow_20251217_110830/
├── logs/
│   ├── launcher.out           # Launcher process stdout
│   ├── launcher.err           # Launcher process stderr (main workflow log)
│   ├── workflow.pid           # Process ID for killing
│   ├── workflow_summary.log   # High-level summary
│   └── errors_consolidated.log # All ERROR-level messages
│
├── download/
│   ├── download_20251217_110903.out
│   └── download_20251217_110903.err
│
├── preprocessing/
│   ├── preprocessing_20251217_110915.out
│   └── preprocessing_20251217_110915.err
│
├── embedding_preparation/
│   └── ...
│
├── embedding_cpu/
│   └── ...
│
├── embedding_gpu/
│   └── ...
│
└── dataset_creation/
    └── ...
```

### Monitoring a Running Workflow

```bash
# Watch the main workflow log
tail -f outputs/2025-12-17/workflow_*/logs/launcher.err

# Check current step
ls -la outputs/2025-12-17/workflow_*/*/

# Check for errors
cat outputs/2025-12-17/workflow_*/logs/errors_consolidated.log
```

### Workflow Summary

After completion, check `workflow_summary.log`:

```
================================================================================
WORKFLOW SUMMARY
================================================================================
Workflow ID: workflow_20251217_110830
Dataset: my_dataset
Status: COMPLETED

Steps:
  download:              ✓ SUCCESS (2.3s)
  preprocessing:         ✓ SUCCESS (45.2s)
  embedding_preparation: SKIPPED
  embedding_cpu:         ✓ SUCCESS (120.5s)
  embedding_gpu:         ✓ SUCCESS (89.3s)
  dataset_creation:      ✓ SUCCESS (15.8s)

Total time: 273.1s
================================================================================
```

---

## Job Cancellation

### Killing a Workflow

When you kill the main workflow process, **all child processes and remote jobs are terminated**:

```bash
# Option 1: Use the PID directly
kill 12345

# Option 2: Use the PID file
kill $(cat outputs/2025-12-17/workflow_*/logs/workflow.pid)
```

### What Happens on Kill

1. **Signal caught** - WorkflowRunner receives SIGTERM/SIGINT
2. **Cleanup initiated** - `_cleanup_all()` is called
3. **Local processes terminated** - Any running subprocess is killed
4. **Remote jobs cancelled** - `scancel` is called via SSH for SLURM jobs
5. **Process exits** - Main workflow process terminates

Log output on termination:

```
Received signal 15, terminating workflow...
Cleaning up all running processes and jobs...
Terminating executor for local...
Terminating local process 12456
Terminating executor for cpu...
Cancelling SLURM job 789012 on cpu
Successfully cancelled job 789012
Cleanup complete
```

### Cancelling Only Remote Jobs

If you need to cancel a specific remote job without stopping the workflow:

```bash
# On the cluster
scancel <job_id>

# Or via SSH
ssh cpu_cluster "scancel <job_id>"
```

⚠️ **Note:** This may cause the workflow to fail when it detects the cancelled job.

---

## Troubleshooting

### Common Issues

#### "SSH connection timed out"

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

#### "Config file not synchronized"

**Cause:** Local and remote configs differ

**Solution:**

```bash
# Option 1: Use --force to skip check
python scripts/workflow/submit_workflow.py --config my_dataset --force

# Option 2: Sync configs manually
ssh cpu_cluster "cd /path/to/project && git pull"
```

#### "All steps run locally - skipping remote config sync check"

**This is expected** when all your steps have `execution_location: local`. No remote validation is needed.

#### "Data transfer failed"

**Cause:** rsync or SSH issue

**Solution:**

```bash
# Test rsync manually
rsync -avz --progress local_file cpu_cluster:/path/

# Check disk space on target
ssh cpu_cluster "df -h /scratch"
```

#### "SLURM job failed"

**Cause:** Resource limits, missing dependencies, or script error

**Solution:**

```bash
# Check job status
ssh cpu_cluster "sacct -j <job_id> --format=JobID,State,ExitCode,MaxRSS"

# Check job output
cat outputs/*/step_name/*.err
```

#### "Process not terminating"

**Cause:** Child process ignoring SIGTERM

**Solution:**

```bash
# Force kill
kill -9 <pid>

# Kill entire process group
kill -9 -<pgid>
```

### Debug Mode

For more verbose output, run in foreground:

```bash
python scripts/workflow/submit_workflow.py --config my_dataset --foreground
```

### Getting Help

1. **Check launcher log:** `outputs/*/logs/launcher.err`
2. **Check step logs:** `outputs/*/step_name/*.err`
3. **Check consolidated errors:** `outputs/*/logs/errors_consolidated.log`
4. **Check workflow summary:** `outputs/*/logs/workflow_summary.log`

For step-specific issues, see the individual step README files:

- [Download README](../download/README.md)
- [Preprocessing README](../preprocessing/README.md)
- [Embedding README](../embed/README.md)
- [Dataset Creation README](../dataset_creation/README.md)

---

## Summary

The unified workflow orchestrator provides:

✅ **Per-step execution locations** - Run each step where it makes sense
✅ **Automatic data transfer** - Seamless movement between locations
✅ **Centralized logging** - All logs in one place
✅ **Graceful termination** - Kill propagates to all processes and remote jobs
✅ **Simple interface** - Single submission script for all modes

**Key commands:**

```bash
# Submit workflow
python scripts/workflow/submit_workflow.py --config my_dataset

# Run in foreground
python scripts/workflow/submit_workflow.py --config my_dataset --foreground

# Kill workflow
kill $(cat outputs/*/logs/workflow.pid)
```
