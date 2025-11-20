#!/bin/bash
# Quick script to find and optionally kill orphaned workflow processes

echo "=== Finding orphaned workflow processes ==="
echo ""

# Find all workflow-related processes
echo "All workflow-related processes:"
ps aux | grep -E "(download_dataset|preprocess|embed|create_ds|run_workflow_master)" | grep -v grep | grep -v "$0"
echo ""

# Find processes with their parent PIDs
echo "Processes with parent PIDs (PPID=1 means orphaned):"
ps -eo pid,ppid,etime,command | grep -E "(download_dataset|preprocess|embed|create_ds|run_workflow_master)" | grep -v grep | grep -v "$0"
echo ""

# Find orphaned processes specifically (PPID = 1)
echo "Orphaned processes (parent died, PPID=1):"
orphaned=$(ps -eo pid,ppid,command | awk '$2 == 1 && /(download_dataset|preprocess|embed|create_ds|run_workflow_master)/ {print $1}')
if [ -z "$orphaned" ]; then
    echo "  No orphaned processes found"
else
    for pid in $orphaned; do
        ps -p "$pid" -o pid,ppid,etime,command 2>/dev/null
    done
    echo ""
    echo "To kill these orphaned processes, run:"
    echo "  kill $orphaned"
    echo ""
    echo "Or force kill:"
    echo "  kill -9 $orphaned"
fi
