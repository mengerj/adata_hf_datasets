#!/bin/bash
# Helper script to kill running local workflows
# Usage: ./scripts/workflow/kill_workflow.sh [workflow_id_or_pid]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUTPUTS_DIR="$PROJECT_DIR/outputs"

# Function to get child processes of a PID
get_child_pids() {
    local parent_pid="$1"
    local child_pids=""
    # Use pstree or ps to find children
    if command -v pstree >/dev/null 2>&1; then
        # pstree -p returns PIDs in parentheses
        child_pids=$(pstree -p "$parent_pid" 2>/dev/null | grep -oP '\(\K[0-9]+' | grep -v "^$parent_pid$" || true)
    else
        # Fallback: use ps to find processes with this PPID
        child_pids=$(ps -o pid= --ppid "$parent_pid" 2>/dev/null || true)
    fi
    echo "$child_pids"
}

# Function to kill process and all its children
kill_process_tree() {
    local pid="$1"
    local force="${2:-0}"  # Optional force flag (1 = use kill -9)

    if ! ps -p "$pid" > /dev/null 2>&1; then
        return 1
    fi

    # Get all child processes recursively
    local child_pids=$(get_child_pids "$pid")
    if [ -n "$child_pids" ]; then
        echo "  Killing subprocesses: $child_pids"
        for child_pid in $child_pids; do
            if ps -p "$child_pid" > /dev/null 2>&1; then
                if [ "$force" = "1" ]; then
                    kill -9 "$child_pid" 2>/dev/null || true
                else
                    kill "$child_pid" 2>/dev/null || true
                fi
            fi
        done
        # Wait a bit for children to die
        sleep 1
    fi

    # Kill the parent process
    if ps -p "$pid" > /dev/null 2>&1; then
        if [ "$force" = "1" ]; then
            kill -9 "$pid" 2>/dev/null || true
        else
            kill "$pid" 2>/dev/null || true
        fi
    fi

    return 0
}

# Function to find and kill workflow by PID file
kill_by_pid_file() {
    local pid_file="$1"
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if ps -p "$pid" > /dev/null 2>&1; then
            echo "Found running workflow with PID: $pid"
            echo "Killing process tree (including subprocesses)..."
            kill_process_tree "$pid" 0
            # Wait and force kill if still running
            sleep 2
            if ps -p "$pid" > /dev/null 2>&1; then
                echo "  Force killing remaining processes..."
                kill_process_tree "$pid" 1
            fi
            echo "✓ Workflow killed"
            return 0
        else
            echo "PID $pid from $pid_file is not running (stale PID file)"
            return 1
        fi
    fi
    return 1
}

# Function to find and kill workflow by process name
kill_by_process() {
    local pattern="$1"
    echo "Searching for processes matching: $pattern"
    local pids=$(pgrep -f "$pattern" || true)
    if [ -z "$pids" ]; then
        echo "No running workflows found matching: $pattern"
        return 1
    fi
    for pid in $pids; do
        echo "Found process: $pid"
        echo "Killing process tree (including subprocesses)..."
        kill_process_tree "$pid" 0
        sleep 1
        if ps -p "$pid" > /dev/null 2>&1; then
            kill_process_tree "$pid" 1
        fi
        echo "✓ Process $pid killed"
    done
    return 0
}

# Function to get process name/command
get_process_info() {
    local pid="$1"
    # Try to get command line, fallback to process name
    local cmd=$(ps -p "$pid" -o command= 2>/dev/null | head -1 || echo "")
    if [ -z "$cmd" ]; then
        cmd=$(ps -p "$pid" -o comm= 2>/dev/null | head -1 || echo "unknown")
    fi
    # Extract step name from command if possible
    local step_name=""
    if echo "$cmd" | grep -q "download_dataset.py"; then
        step_name="[download]"
    elif echo "$cmd" | grep -q "preprocess.py"; then
        step_name="[preprocessing]"
    elif echo "$cmd" | grep -q "embed"; then
        step_name="[embedding]"
    elif echo "$cmd" | grep -q "create_ds.py"; then
        step_name="[dataset_creation]"
    fi
    echo "$step_name|$cmd"
}

# Function to list all running workflows
list_workflows() {
    echo "Searching for running local workflows..."
    echo ""

    # Track PIDs we've already found from PID files (space-separated with leading/trailing spaces)
    local found_pids=" "
    local found=0

    # Find all PID files first
    if [ -d "$OUTPUTS_DIR" ]; then
        while IFS= read -r -d '' pid_file; do
            local pid=$(cat "$pid_file" 2>/dev/null || echo "")
            if [ -n "$pid" ] && ps -p "$pid" > /dev/null 2>&1; then
                found=1
                found_pids="$found_pids$pid "
                local workflow_dir=$(dirname "$(dirname "$pid_file")")
                local workflow_id=$(basename "$workflow_dir")
                echo "  PID: $pid | Workflow: $workflow_id (master)"
                echo "    PID file: $pid_file"

                # Find and show child processes
                local child_pids=$(get_child_pids "$pid")
                if [ -n "$child_pids" ]; then
                    echo "    Subprocesses:"
                    for child_pid in $child_pids; do
                        if ps -p "$child_pid" > /dev/null 2>&1; then
                            local proc_info=$(get_process_info "$child_pid")
                            local step_name=$(echo "$proc_info" | cut -d'|' -f1)
                            local cmd=$(echo "$proc_info" | cut -d'|' -f2- | cut -c1-60)
                            echo "      - PID: $child_pid $step_name ($cmd...)"
                        fi
                    done
                fi

                echo "    Kill with: kill $pid (will also kill subprocesses)"
                echo ""
            fi
        done < <(find "$OUTPUTS_DIR" -name "workflow_master.pid" -type f -print0 2>/dev/null || true)
    fi

    # Also search for processes by name (but skip ones we already found)
    local pids=$(pgrep -f "run_workflow_master.py" 2>/dev/null || true)
    if [ -n "$pids" ]; then
        for pid in $pids; do
            # Check if this PID is already in our found_pids list (with word boundaries)
            if ! echo "$found_pids" | grep -qE " $pid |^$pid | $pid$" 2>/dev/null; then
                found=1
                echo "  PID: $pid | Process: run_workflow_master.py (no PID file found)"

                # Find and show child processes
                local child_pids=$(get_child_pids "$pid")
                if [ -n "$child_pids" ]; then
                    echo "    Subprocesses:"
                    for child_pid in $child_pids; do
                        if ps -p "$child_pid" > /dev/null 2>&1; then
                            local proc_info=$(get_process_info "$child_pid")
                            local step_name=$(echo "$proc_info" | cut -d'|' -f1)
                            local cmd=$(echo "$proc_info" | cut -d'|' -f2- | cut -c1-60)
                            echo "      - PID: $child_pid $step_name ($cmd...)"
                        fi
                    done
                fi

                echo "    Kill with: kill $pid (will also kill subprocesses)"
                echo ""
            fi
        done
    fi

    # Also check for orphaned processes (parent died, PPID=1)
    echo "Checking for orphaned processes (parent process died)..."
    local orphaned=$(ps -eo pid,ppid,command 2>/dev/null | awk '$2 == 1 && /(download_dataset|preprocess|embed|create_ds|run_workflow_master)/ {print $1}' || true)
    if [ -n "$orphaned" ]; then
        found=1
        echo ""
        echo "  ⚠️  ORPHANED PROCESSES FOUND (parent process died):"
        for pid in $orphaned; do
            if ps -p "$pid" > /dev/null 2>&1; then
                local proc_info=$(get_process_info "$pid")
                local step_name=$(echo "$proc_info" | cut -d'|' -f1)
                local cmd=$(echo "$proc_info" | cut -d'|' -f2- | cut -c1-60)
                echo "    PID: $pid $step_name (orphaned, PPID=1)"
                echo "      Command: $cmd..."
                echo "      Kill with: kill $pid"
                echo ""
            fi
        done
        echo "  To kill all orphaned processes:"
        echo "    ps -eo pid,ppid,command | awk '\$2 == 1 && /(download_dataset|preprocess|embed|create_ds|run_workflow_master)/ {print \$1}' | xargs kill"
        echo ""
    fi

    if [ $found -eq 0 ]; then
        echo "  No running workflows found"
        echo ""
    fi

    return 0
}

# Main logic
if [ $# -eq 0 ]; then
    # No arguments: list all running workflows
    list_workflows
    echo "To kill a workflow, run:"
    echo "  $0 <pid>              # Kill by PID"
    echo "  $0 <workflow_id>      # Kill by workflow ID (e.g., workflow_local_20251120_091744)"
    echo "  $0 --all              # Kill all running workflows"
    exit 0
fi

arg="$1"

if [ "$arg" == "--all" ] || [ "$arg" == "-a" ]; then
    # Kill all running workflows
    echo "Killing all running workflows..."
    killed=0

    # Kill by PID files
    if [ -d "$OUTPUTS_DIR" ]; then
        while IFS= read -r -d '' pid_file; do
            if kill_by_pid_file "$pid_file"; then
                killed=1
            fi
        done < <(find "$OUTPUTS_DIR" -name "workflow_master.pid" -type f -print0 2>/dev/null || true)
    fi

    # Kill by process name (for workflows without PID files)
    # Get all PIDs first, then kill each with its tree
    local pids=$(pgrep -f "run_workflow_master.py" 2>/dev/null || true)
    if [ -n "$pids" ]; then
        for pid in $pids; do
            echo "Killing workflow process tree: $pid"
            kill_process_tree "$pid" 0
            sleep 1
            if ps -p "$pid" > /dev/null 2>&1; then
                kill_process_tree "$pid" 1
            fi
            killed=1
        done
    fi

    # Also kill orphaned processes (parent died, PPID=1)
    echo ""
    echo "Checking for orphaned processes..."
    local orphaned=$(ps -eo pid,ppid,command 2>/dev/null | awk '$2 == 1 && /(download_dataset|preprocess|embed|create_ds|run_workflow_master)/ {print $1}' || true)
    if [ -n "$orphaned" ]; then
        for pid in $orphaned; do
            if ps -p "$pid" > /dev/null 2>&1; then
                echo "Killing orphaned process: $pid"
                kill "$pid" 2>/dev/null || kill -9 "$pid" 2>/dev/null
                killed=1
            fi
        done
    fi

    if [ $killed -eq 0 ]; then
        echo "No running workflows found to kill"
        exit 1
    fi

    echo "✓ All workflows and orphaned processes killed"
    exit 0
fi

# Check if argument is a PID (numeric)
if [[ "$arg" =~ ^[0-9]+$ ]]; then
    echo "Killing workflow with PID: $arg (and all subprocesses)"
    if ps -p "$arg" > /dev/null 2>&1; then
        kill_process_tree "$arg" 0
        # Wait and force kill if still running
        sleep 2
        if ps -p "$arg" > /dev/null 2>&1; then
            echo "  Force killing remaining processes..."
            kill_process_tree "$arg" 1
        fi
        echo "✓ Workflow killed"
    else
        echo "Error: PID $arg is not running"
        exit 1
    fi
    exit 0
fi

# Check if argument is a workflow directory name
if [ -d "$OUTPUTS_DIR" ]; then
    # Try to find workflow directory
    workflow_dir=$(find "$OUTPUTS_DIR" -type d -name "*$arg*" | head -1)
    if [ -n "$workflow_dir" ]; then
        pid_file="$workflow_dir/logs/workflow_master.pid"
        if kill_by_pid_file "$pid_file"; then
            exit 0
        fi
    fi
fi

# If we get here, try to kill by process name pattern
if kill_by_process "$arg"; then
    exit 0
fi

echo "Error: Could not find workflow matching: $arg"
echo "Run '$0' without arguments to list all running workflows"
exit 1
