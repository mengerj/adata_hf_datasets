#!/usr/bin/env python3
"""
Test the waiting logic from run_embed_new.slurm to identify failure points.
"""

import subprocess
import tempfile
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_sacct_commands():
    """Test sacct commands that might be causing issues."""
    logger.info("=== Testing sacct Commands ===")

    # Test with real job IDs from your example
    test_job_ids = ["7517247", "7517248", "999999"]  # Include a non-existent job

    for job_id in test_job_ids:
        logger.info(f"\nTesting job ID: {job_id}")

        # Test the exact command from the script
        cmd = f"sacct -j {job_id} --format=JobID,State --noheader --parsable2 2>/dev/null | grep '{job_id}_' | cut -d'|' -f2"
        logger.info(f"Command: {cmd}")

        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=10
            )

            logger.info(f"Exit code: {result.returncode}")
            logger.info(f"Stdout: '{result.stdout}'")
            logger.info(f"Stderr: '{result.stderr}'")

            if result.returncode != 0:
                logger.warning(f"Command failed with exit code {result.returncode}")

            # Test the state parsing
            if result.stdout.strip():
                states = result.stdout.strip().split("\n")
                logger.info(f"Parsed states: {states}")

                # Test the running count logic
                running_count = 0
                for state in states:
                    if state.strip() in ["PENDING", "RUNNING", "COMPLETING"]:
                        running_count += 1

                logger.info(f"Running count: {running_count}")
            else:
                logger.info("No output - job might be completed or not found")

        except subprocess.TimeoutExpired:
            logger.error(f"Command timed out for job {job_id}")
        except Exception as e:
            logger.error(f"Command failed for job {job_id}: {e}")


def test_job_file_reading():
    """Test the job file reading logic."""
    logger.info("=== Testing Job File Reading ===")

    # Create a temporary job file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        job_file = f.name
        f.write("7517247\n")
        f.write("7517248\n")
        f.write("\n")  # Empty line
        f.write("invalid_job_id\n")
        f.write("123456\n")

    try:
        logger.info(f"Created test job file: {job_file}")

        # Test the reading logic from the script
        with open(job_file, "r") as f:
            for line_num, line in enumerate(f, 1):
                array_job_id = line.strip()
                logger.info(f"Line {line_num}: '{line.strip()}'")

                # Test the validation logic
                if array_job_id and array_job_id.isdigit():
                    logger.info(f"  ✓ Valid job ID: {array_job_id}")
                else:
                    logger.info(f"  ✗ Invalid job ID: '{array_job_id}'")

    finally:
        os.unlink(job_file)


def test_bash_error_conditions():
    """Test bash conditions that might cause set -euo pipefail to fail."""
    logger.info("=== Testing Bash Error Conditions ===")

    # Test commands that might return non-zero exit codes
    test_commands = [
        "echo 'test' | grep 'notfound'",  # grep returns 1 when no match
        "sacct -j 999999 --format=JobID,State --noheader --parsable2 2>/dev/null",  # Might return non-zero
        "echo '' | wc -l",  # Empty input
        "echo 'RUNNING\nCOMPLETED' | grep -E '^(PENDING|RUNNING|COMPLETING)$' | wc -l",  # Should work
        "echo 'COMPLETED\nFAILED' | grep -E '^(PENDING|RUNNING|COMPLETING)$' | wc -l",  # Should return 0
    ]

    for cmd in test_commands:
        logger.info(f"\nTesting: {cmd}")

        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=5
            )

            logger.info(f"Exit code: {result.returncode}")
            logger.info(f"Stdout: '{result.stdout.strip()}'")

            if result.returncode != 0:
                logger.warning(
                    f"⚠️  Command returned non-zero exit code: {result.returncode}"
                )
                logger.warning("This could cause 'set -euo pipefail' to fail")

        except Exception as e:
            logger.error(f"Command failed: {e}")


def simulate_waiting_logic():
    """Simulate the exact waiting logic from the script."""
    logger.info("=== Simulating Waiting Logic ===")

    # Create a test job file with real job IDs
    test_job_ids = ["7517247", "7517248"]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        job_file = f.name
        for job_id in test_job_ids:
            f.write(f"{job_id}\n")

    try:
        logger.info(f"Simulating with job file: {job_file}")

        # Simulate the bash script logic in Python
        with open(job_file, "r") as f:
            for line in f:
                array_job_id = line.strip()

                if array_job_id and array_job_id.isdigit():
                    logger.info(f"Processing job {array_job_id}")

                    # Simulate the sacct command
                    cmd = [
                        "sacct",
                        "-j",
                        array_job_id,
                        "--format=JobID,State",
                        "--noheader",
                        "--parsable2",
                    ]

                    try:
                        result = subprocess.run(
                            cmd, capture_output=True, text=True, timeout=10
                        )

                        logger.info(f"sacct exit code: {result.returncode}")

                        if result.returncode == 0 and result.stdout:
                            # Parse the output
                            lines = result.stdout.strip().split("\n")
                            array_tasks = [
                                line for line in lines if f"{array_job_id}_" in line
                            ]

                            logger.info(f"Found {len(array_tasks)} array tasks")

                            if array_tasks:
                                states = [
                                    line.split("|")[1]
                                    for line in array_tasks
                                    if "|" in line
                                ]
                                logger.info(f"States: {states}")

                                running_states = [
                                    s
                                    for s in states
                                    if s in ["PENDING", "RUNNING", "COMPLETING"]
                                ]
                                completed_states = [
                                    s
                                    for s in states
                                    if s in ["COMPLETED", "COMPLETED+"]
                                ]
                                failed_states = [
                                    s
                                    for s in states
                                    if s in ["FAILED", "CANCELLED", "TIMEOUT"]
                                ]

                                logger.info(
                                    f"Running: {len(running_states)}, Completed: {len(completed_states)}, Failed: {len(failed_states)}"
                                )
                            else:
                                logger.warning(
                                    f"No array tasks found for job {array_job_id}"
                                )
                        else:
                            logger.warning("sacct command failed or returned no output")

                    except Exception as e:
                        logger.error(f"Error running sacct for job {array_job_id}: {e}")

    finally:
        os.unlink(job_file)


def main():
    """Run all tests."""
    tests = [
        test_bash_error_conditions,
        test_job_file_reading,
        test_sacct_commands,
        simulate_waiting_logic,
    ]

    for test_func in tests:
        logger.info(f"\n{'=' * 60}")
        try:
            test_func()
        except Exception as e:
            logger.error(f"Test {test_func.__name__} failed: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    main()
