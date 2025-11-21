# How to Stop/Kill Local Workflows

This guide explains how to stop running local workflows, both for newly launched workflows (with PID tracking) and older workflows launched before this feature was implemented.

## Quick Reference

### For New Workflows (with PID tracking)

When you submit a workflow in background mode, you'll see output like:

```
================================================================================
WORKFLOW RUNNING IN BACKGROUND
================================================================================
Process ID (PID): 12345
To stop this workflow, run: kill 12345
Or use: kill $(cat outputs/2025-11-20/workflow_local_20251120_091744/logs/workflow_master.pid)
PID file: outputs/2025-11-20/workflow_local_20251120_091744/logs/workflow_master.pid
================================================================================
```

**To kill the workflow:**

```bash
# Option 1: Use the PID directly
kill 12345

# Option 2: Use the PID file
kill $(cat outputs/2025-11-20/workflow_local_20251120_091744/logs/workflow_master.pid)

# Option 3: Use the helper script
./scripts/workflow/kill_workflow.sh 12345
```

### For Old Workflows (launched before PID tracking)

If you launched a workflow before this feature was implemented, you need to find the process manually:

**Method 1: Find by process name**

```bash
# Find the PID
pgrep -f "run_workflow_master.py"

# Kill it
pkill -f "run_workflow_master.py"

# Or kill a specific PID
kill <PID>
```

**Method 2: Use the helper script**

```bash
# List all running workflows
./scripts/workflow/kill_workflow.sh

# Kill all running workflows
./scripts/workflow/kill_workflow.sh --all
```

**Method 3: Find by workflow directory**

```bash
# If you know the workflow ID (from the output directory name)
./scripts/workflow/kill_workflow.sh workflow_local_20251120_091744
```

## Detailed Methods

### Method 1: Using the Helper Script (Recommended)

The helper script `scripts/workflow/kill_workflow.sh` provides the easiest way to manage workflows:

```bash
# List all running workflows
./scripts/workflow/kill_workflow.sh

# Kill a specific workflow by PID
./scripts/workflow/kill_workflow.sh 12345

# Kill a specific workflow by workflow ID
./scripts/workflow/kill_workflow.sh workflow_local_20251120_091744

# Kill all running workflows
./scripts/workflow/kill_workflow.sh --all
```

### Method 2: Manual Process Finding

**Step 1: Find the process**

```bash
# Find processes matching the workflow master script
ps aux | grep run_workflow_master.py

# Or use pgrep for just the PID
pgrep -f "run_workflow_master.py"
```

**Step 2: Kill the process**

```bash
# Graceful kill (SIGTERM)
kill <PID>

# Force kill if graceful doesn't work (SIGKILL)
kill -9 <PID>
```

### Method 3: Using PID Files (New Workflows Only)

For workflows launched with PID tracking:

```bash
# Find all PID files
find outputs -name "workflow_master.pid" -type f

# Check if a PID is still running
ps -p $(cat outputs/2025-11-20/workflow_local_20251120_091744/logs/workflow_master.pid)

# Kill using the PID file
kill $(cat outputs/2025-11-20/workflow_local_20251120_091744/logs/workflow_master.pid)
```

## Understanding Process Hierarchy

When you launch a workflow in background mode, the process hierarchy looks like:

```
nohup (PID: 12345)
  └── caffeinate (if available)
      └── python run_workflow_master.py
          └── (subprocesses for each step)
```

Killing the parent process (PID 12345) will terminate all child processes. The workflow uses `start_new_session=True`, which means child processes are in the same process group and will be terminated when the parent is killed.

## Troubleshooting

### Process won't die with `kill`

If a normal `kill` doesn't work, use force kill:

```bash
kill -9 <PID>
```

### Multiple workflows running

To see all running workflows:

```bash
./scripts/workflow/kill_workflow.sh
```

To kill all at once:

```bash
./scripts/workflow/kill_workflow.sh --all
```

### Stale PID files

If a PID file exists but the process is not running, the PID file is stale. You can safely ignore it or delete it:

```bash
rm outputs/2025-11-20/workflow_local_20251120_091744/logs/workflow_master.pid
```

### Finding workflows by output directory

If you know the workflow output directory but not the PID:

```bash
# The workflow ID is in the directory name
ls outputs/2025-11-20/

# Use the workflow ID with the helper script
./scripts/workflow/kill_workflow.sh workflow_local_20251120_091744
```

## Best Practices

1. **Always use the PID file method for new workflows** - it's the most reliable
2. **Use the helper script** - it handles edge cases and provides better error messages
3. **Check if the process is actually running** before killing - use `ps -p <PID>` or the helper script's list function
4. **Use graceful kill first** - `kill <PID>` sends SIGTERM, which allows processes to clean up. Only use `kill -9` if necessary.

## Examples

### Example 1: Kill a workflow you just launched

```bash
# You see this in the output:
# Process ID (PID): 12345
# To stop this workflow, run: kill 12345

kill 12345
```

### Example 2: Find and kill an old workflow

```bash
# Find the process
pgrep -f "run_workflow_master.py"
# Output: 67890

# Kill it
kill 67890
```

### Example 3: Kill all workflows

```bash
./scripts/workflow/kill_workflow.sh --all
```

### Example 4: List workflows to find the right one

```bash
./scripts/workflow/kill_workflow.sh
# Output:
#   PID: 12345 | Workflow: workflow_local_20251120_091744
#     PID file: outputs/2025-11-20/workflow_local_20251120_091744/logs/workflow_master.pid
#     Kill with: kill 12345
#
#   PID: 67890 | Process: run_workflow_master.py (no PID file found)
#     Kill with: kill 67890

# Kill the specific one you want
kill 12345
```
