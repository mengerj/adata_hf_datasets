import logging
import os
import platform
import threading
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import psutil
import pynvml
import torch


class SystemMonitor:
    """A class for monitoring system resource usage over time.

    Parameters
    ----------
    interval : int, optional
        The interval in seconds between each monitoring update. Default is 1 second.
    gpu_idx : int or list of int, optional
        The index (or list of indices) of the GPU(s) to monitor. Default is None.
    logger : logging.Logger, optional
        A logger instance to use for logging messages. If not provided, a logger is created.
    """

    def __init__(self, interval=1, gpu_idx=None, logger=None):
        self.interval = interval
        self.gpu_indices = [gpu_idx] if isinstance(gpu_idx, int) else gpu_idx
        self.num_cpus = psutil.cpu_count(logical=True)
        self.cpu_usage = []      # List of tuples (timestamp, total_cpu_usage_percent)
        self.cpu_per_core = []   # List of tuples (timestamp, avg_cpu_usage_per_core)
        self.memory_usage = []   # List of tuples (timestamp, used_memory_gb relative to baseline)
        self.disk_io = []        # List of tuples (timestamp, read_rate_mb_s, write_rate_mb_s)
        self.total_memory = psutil.virtual_memory().total / (1024**3)  # GB
        self.baseline_memory = psutil.virtual_memory().used / (1024**3)  # GB
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._monitor)
        self.logger = logger or logging.getLogger(__name__)
        self.num_threads = []
        # self.cpu_affinity = []

        # GPU Monitoring Initialization
        self.gpu_available = False
        self.gpu_usage = []         # For each GPU, list of tuples (timestamp, usage_percent)
        self.gpu_memory_usage = []  # For each GPU, list of tuples (timestamp, used_memory_gb)
        self.gpu_name = None

        self._initialize_gpu_monitoring()

        # A list to store event markers for logging usage events.
        # Each event is a dict with keys: 'timestamp' and 'message'
        self.events = []

    def _initialize_gpu_monitoring(self):
        # Try NVIDIA GPU
        try:
            import pynvml

            pynvml.nvmlInit()
            self.gpu_handles = []
            self.gpu_names = []

            # First check CUDA_VISIBLE_DEVICES
            assigned_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", None)
            if assigned_gpus:
                gpu_entries = assigned_gpus.split(",")
                for entry in gpu_entries:
                    entry = entry.strip()
                    try:
                        gpu_idx = int(entry)
                        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_idx)
                        self.gpu_handles.append(handle)
                    except ValueError:
                        handle = pynvml.nvmlDeviceGetHandleByUUID(entry)
            else:
                if self.gpu_indices is not None:
                    for idx in self.gpu_indices:
                        handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
                        self.gpu_handles.append(handle)
                else:
                    device_count = pynvml.nvmlDeviceGetCount()
                    for idx in range(device_count):
                        handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
                        self.gpu_handles.append(handle)

            for handle in self.gpu_handles:
                name = pynvml.nvmlDeviceGetName(handle)
                self.gpu_names.append(name.decode() if isinstance(name, bytes) else name)

            self.gpu_total_memory = [
                pynvml.nvmlDeviceGetMemoryInfo(handle).total / (1024**3) for handle in self.gpu_handles
            ]
            self.gpu_available = True
            self.gpu_type = "NVIDIA"
            self.logger.info(f"Detected NVIDIA GPUs: {self.gpu_names}")

        except Exception as e:
            self.logger.info(f"No NVIDIA GPU detected or pynvml not installed: {str(e)}")
            if platform.system() == "Darwin" and "macOS" in platform.platform():
                self.gpu_available = False
                self.gpu_type = "Apple"
                self.gpu_name = "Apple Integrated GPU"
                self.logger.info("Detected Apple GPU. But not supported for detailed monitoring.")
            else:
                self.logger.info("No supported GPU detected.")
                self.gpu_available = False

    def _monitor(self):
        process = psutil.Process()
        prev_disk_io_counters = None
        prev_time = time.time()
        while not self._stop_event.is_set():
            timestamp = time.time()
            interval_duration = timestamp - prev_time
            prev_time = timestamp

            # Measure CPU usage per core
            cpu_percents = psutil.cpu_percent(interval=self.interval, percpu=True)
            total_cpu_usage_percent = sum(cpu_percents)
            total_cpu_usage_cores = total_cpu_usage_percent / self.num_cpus
            self.cpu_usage.append((timestamp, total_cpu_usage_percent))
            self.cpu_per_core.append((timestamp, total_cpu_usage_cores))

            # Measure memory usage
            mem = psutil.virtual_memory()
            used_memory_gb = (mem.total - mem.available) / (1024**3)
            used_memory_gb -= self.baseline_memory
            self.memory_usage.append((timestamp, used_memory_gb))

            # Measure disk I/O
            disk_io_counters = psutil.disk_io_counters()
            if prev_disk_io_counters is not None:
                read_bytes = disk_io_counters.read_bytes - prev_disk_io_counters.read_bytes
                write_bytes = disk_io_counters.write_bytes - prev_disk_io_counters.write_bytes
                read_rate_mb_s = (read_bytes / (1024**2)) / interval_duration if interval_duration > 0 else 0
                write_rate_mb_s = (write_bytes / (1024**2)) / interval_duration if interval_duration > 0 else 0
            else:
                read_rate_mb_s = write_rate_mb_s = 0

            self.disk_io.append((timestamp, read_rate_mb_s, write_rate_mb_s))
            prev_disk_io_counters = disk_io_counters

            # Get number of threads
            self.num_threads.append((timestamp, process.num_threads()))
            # self.cpu_affinity.append((timestamp, process.cpu_affinity()))

            # GPU Monitoring
            if self.gpu_available:
                if self.gpu_type == "NVIDIA":
                    self._monitor_nvidia_gpu(timestamp)
                elif self.gpu_type == "Apple":
                    self._monitor_apple_gpu(timestamp)
                else:
                    pass  # Unsupported GPU type

    def _monitor_nvidia_gpu(self, timestamp):
        try:
            for idx, handle in enumerate(self.gpu_handles):
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_usage_percent = util.gpu
                gpu_memory_used_gb = mem_info.used / (1024**3)
                if idx >= len(self.gpu_usage):
                    self.gpu_usage.append([])
                    self.gpu_memory_usage.append([])
                self.gpu_usage[idx].append((timestamp, gpu_usage_percent))
                self.gpu_memory_usage[idx].append((timestamp, gpu_memory_used_gb))
        except Exception as e:
            self.logger.error(f"Error monitoring NVIDIA GPU: {e}")

    def _monitor_apple_gpu(self, timestamp):
        self.gpu_usage.append((timestamp, None))
        self.gpu_memory_usage.append((timestamp, None))

    def start(self):
        """Start monitoring system resources."""
        self._thread.start()

    def stop(self):
        """Stop monitoring system resources."""
        self._stop_event.set()
        self._thread.join()
        if self.gpu_available and self.gpu_type == "NVIDIA":
            import pynvml
            pynvml.nvmlShutdown()

    def summarize(self):
        """
        Summarizes the collected metrics.

        Returns
        -------
        summary : dict
            A dictionary containing the mean and maximum usage for CPU, memory, disk I/O, and GPU (if available).
        """
        summary = {}

        # CPU Usage
        total_cpu_usages = [usage for _, usage in self.cpu_usage]
        summary["cpu_usage_mean"] = sum(total_cpu_usages) / len(total_cpu_usages)
        summary["cpu_usage_max"] = max(total_cpu_usages)

        # Core Utilization
        core_utilizations = [cores for _, cores in self.cpu_per_core]
        summary["core_usage_mean"] = sum(core_utilizations) / len(core_utilizations)
        summary["core_usage_max"] = max(core_utilizations)

        # Memory Usage
        memory_usages = [usage for _, usage in self.memory_usage]
        summary["memory_usage_mean"] = sum(memory_usages) / len(memory_usages)
        summary["memory_usage_max"] = max(memory_usages)
        summary["total_memory"] = self.total_memory
        summary["baseline_memory"] = self.baseline_memory

        # Disk I/O
        read_rates = [read for _, read, _ in self.disk_io]
        write_rates = [write for _, _, write in self.disk_io]
        summary["disk_read_mb_s_mean"] = sum(read_rates) / len(read_rates)
        summary["disk_read_mb_s_max"] = max(read_rates)
        summary["disk_write_mb_s_mean"] = sum(write_rates) / len(write_rates)
        summary["disk_write_mb_s_max"] = max(write_rates)

        # GPU Usage
        if self.gpu_available:
            summary["total_gpu_memory"] = sum(self.gpu_total_memory)
            if self.gpu_type == "NVIDIA":
                summary["gpu_metrics"] = []
                for idx, (usage_data, memory_data, name, total_memory) in enumerate(
                    zip(self.gpu_usage, self.gpu_memory_usage, self.gpu_names, self.gpu_total_memory, strict=False)
                ):
                    gpu_summary = {}
                    gpu_usages = [usage for _, usage in usage_data if usage is not None]
                    gpu_memory_usages = [usage for _, usage in memory_data if usage is not None]

                    if gpu_usages:
                        gpu_summary["usage_mean"] = sum(gpu_usages) / len(gpu_usages)
                        gpu_summary["usage_max"] = max(gpu_usages)
                    if gpu_memory_usages:
                        gpu_summary["memory_usage_mean"] = sum(gpu_memory_usages) / len(gpu_memory_usages)
                        gpu_summary["memory_usage_max"] = max(gpu_memory_usages)

                    gpu_summary["name"] = name.decode() if isinstance(name, bytes) else name
                    gpu_summary["total_memory"] = total_memory
                    gpu_summary["gpu_id"] = idx

                    summary["gpu_metrics"].append(gpu_summary)
            else:
                summary["gpu_metrics"] = [
                    {
                        "name": self.gpu_name,
                        "usage_mean": None,
                        "usage_max": None,
                        "memory_usage_mean": None,
                        "memory_usage_max": None,
                        "gpu_id": 0,
                    }
                ]
        else:
            summary["gpu_metrics"] = []

        return summary

    def print_summary(self):
        """Prints a formatted summary of the metrics."""
        summary = self.summarize()
        print("\nSystem Resource Usage Summary:")
        print(
            f"Core Utilization (mean/max % per core): {summary['core_usage_mean']:.2f}/{summary['core_usage_max']:.2f}% on {self.num_cpus} cores"
        )
        print(f"Memory Usage (mean/max GB): {summary['memory_usage_mean']:.2f}/{summary['memory_usage_max']:.2f} GB")
        print(f"Total System Memory: {summary['total_memory']:.2f} GB")
        print("Baseline Memory Usage: {:.2f} GB".format(summary["baseline_memory"]))
        print(
            f"Disk Read Rate (mean/max MB/s): {summary['disk_read_mb_s_mean']:.2f}/{summary['disk_read_mb_s_max']:.2f} MB/s"
        )
        print(
            f"Disk Write Rate (mean/max MB/s): {summary['disk_write_mb_s_mean']:.2f}/{summary['disk_write_mb_s_max']:.2f} MB/s"
        )
        if self.gpu_available:
            print("\nGPU Metrics:")
            print("Total GPU Memory: {:.2f} GB".format(summary["total_gpu_memory"]))
            for gpu in summary["gpu_metrics"]:
                print(f"\nGPU {gpu['gpu_id']}: {gpu['name']}")
                if gpu.get("usage_mean") is not None:
                    print(f"  Usage (mean/max %): {gpu['usage_mean']:.2f}/{gpu['usage_max']:.2f}%")
                if gpu.get("memory_usage_mean") is not None:
                    print(
                        f"  Memory Usage (mean/max GB): {gpu['memory_usage_mean']:.2f}/{gpu['memory_usage_max']:.2f} GB"
                    )
                    print(f"  Total Memory: {gpu['total_memory']:.2f} GB")
        else:
            print("\nNo supported GPU detected.")

    def log_event(self, message):
        """
        Log an event with current usage details and record it for plotting.

        This function logs the current resource usage along with the mean and maximum usage observed
        up to that point for CPU, memory, GPU usage, and GPU memory usage (if available).
        It also stores the event (timestamp and message) in the monitor, which will be later annotated on the plots.

        Parameters
        ----------
        message : str
            A descriptive message for the event.

        Notes
        -----
        CPU and memory usage values are computed from the collected data up to this call.
        GPU metrics are computed if GPU monitoring is enabled.
        Data sources:
          - CPU and memory metrics are gathered from psutil.
          - GPU metrics are collected via pynvml.
        """
        current_time = time.time()
        event = {"timestamp": current_time, "message": message}
        self.events.append(event)

        # CPU Usage Stats
        if self.cpu_usage:
            cpu_values = [usage for ts, usage in self.cpu_usage]
            current_cpu = self.cpu_usage[-1][1]
            mean_cpu = sum(cpu_values) / len(cpu_values)
            max_cpu = max(cpu_values)
        else:
            current_cpu = mean_cpu = max_cpu = None

        # Memory Usage Stats
        if self.memory_usage:
            mem_values = [usage for ts, usage in self.memory_usage]
            current_mem = self.memory_usage[-1][1]
            mean_mem = sum(mem_values) / len(mem_values)
            max_mem = max(mem_values)
        else:
            current_mem = mean_mem = max_mem = None

        # GPU Metrics Stats (if available)
        gpu_message = ""
        if self.gpu_available and self.gpu_type == "NVIDIA":
            for idx, usage_data in enumerate(self.gpu_usage):
                if usage_data:
                    gpu_usages = [usage for ts, usage in usage_data if usage is not None]
                    current_gpu = usage_data[-1][1] if gpu_usages else None
                    mean_gpu = sum(gpu_usages) / len(gpu_usages) if gpu_usages else None
                    max_gpu = max(gpu_usages) if gpu_usages else None

                    memory_data = self.gpu_memory_usage[idx]
                    if memory_data:
                        gpu_mem_usages = [mem for ts, mem in memory_data if mem is not None]
                        current_gpu_mem = memory_data[-1][1] if gpu_mem_usages else None
                        mean_gpu_mem = sum(gpu_mem_usages) / len(gpu_mem_usages) if gpu_mem_usages else None
                        max_gpu_mem = max(gpu_mem_usages) if gpu_mem_usages else None
                    else:
                        current_gpu_mem = mean_gpu_mem = max_gpu_mem = None

                    gpu_message += (
                        f" | GPU {idx} ({self.gpu_names[idx]}): "
                        f"Usage (current/mean/max %): {current_gpu:.2f}/{mean_gpu:.2f}/{max_gpu:.2f}, "
                        f"Memory (current/mean/max GB): {current_gpu_mem:.2f}/{mean_gpu_mem:.2f}/{max_gpu_mem:.2f}"
                    )
        elif self.gpu_available:
            gpu_message = " | GPU metrics not available for this GPU type."

        self.logger.info(
            f"Event logged at {time.strftime('%H:%M:%S', time.localtime(current_time))}: {message}. "
            f"CPU Usage (current/mean/max %): {current_cpu:.2f}/{mean_cpu:.2f}/{max_cpu:.2f} | "
            f"Memory Usage (current/mean/max GB): {current_mem:.2f}/{mean_mem:.2f}/{max_mem:.2f}"
            f"{gpu_message}"
        )

    def save(self, save_dir):
        """Save the metrics as a CSV file.

        Parameters
        ----------
        save_dir : str
            Directory path to save the CSV file.
        """
        import pandas as pd

        name = "sys_metrics.csv"
        save_path = os.path.join(save_dir, name)
        summary = self.summarize()
        df = pd.DataFrame([summary])
        df.to_csv(save_path, index=False)
        self.logger.info(f"Metrics saved to {save_path}")

    def plot_metrics(self, save_dir=None):
        """
        Plots the collected metrics over time and annotates any logged events.

        If save_dir is provided, the plots are saved to the specified directory.

        Parameters
        ----------
        save_dir : str, optional
            Directory path to save the plots. If None, the plots are shown.
        """
        time_format = "%H:%M:%S"

        def format_time_ticks(timestamps):
            """Helper function to format x-axis ticks based on timestamps."""
            num_points = len(timestamps)
            max_labels = 10  # Maximum number of x-axis labels
            if num_points <= max_labels:
                tick_positions = range(num_points)
                tick_labels = [time.strftime(time_format, time.localtime(ts)) for ts in timestamps]
            else:
                tick_positions = np.linspace(0, num_points - 1, max_labels, dtype=int)
                tick_labels = [time.strftime(time_format, time.localtime(timestamps[pos])) for pos in tick_positions]
            return tick_positions, tick_labels

        # For plotting events, we define a helper to get relative times given a list of (timestamp, value) pairs.
        def get_relative_times(data):
            times = [t for t, _ in data]
            base = times[0] if times else 0
            return [t - base for t in times]

        # --- CPU Usage Plot ---
        if self.cpu_per_core:
            timestamps, cpu_usages = zip(*self.cpu_per_core, strict=False)
            rel_times = get_relative_times(self.cpu_per_core)
            tick_positions, tick_labels = format_time_ticks(timestamps)
            plt.figure()
            plt.plot(rel_times, cpu_usages, label="Avg. CPU % per core")
            # Annotate events
            for i, event in enumerate(self.events, start=1):
                # Find relative time closest to event timestamp
                event_rel = event["timestamp"] - timestamps[0]
                plt.axvline(x=event_rel, color='red', linestyle='--', alpha=0.7)
                plt.text(event_rel, max(cpu_usages)*0.95, f"t{i}", rotation=90, verticalalignment='top', color='red')
            plt.xlabel("Time (s)")
            plt.ylabel("CPU Usage (% per core)")
            plt.title(f"Avg. CPU Usage Over Time ({self.num_cpus} cores)")
            plt.xticks(tick_positions, tick_labels, rotation=45)
            plt.ylim(0, 100)
            plt.tight_layout()
            if save_dir:
                plt.savefig(os.path.join(save_dir, "cpu_usage.png"))
                plt.close()
            else:
                plt.legend()
                plt.show()

        # --- Memory Usage Plot ---
        if self.memory_usage:
            timestamps, mem_usages = zip(*self.memory_usage, strict=False)
            rel_times = get_relative_times(self.memory_usage)
            tick_positions, tick_labels = format_time_ticks(timestamps)
            plt.figure()
            plt.plot(rel_times, mem_usages, label="Memory Usage (GB)")
            # Annotate events
            for i, event in enumerate(self.events, start=1):
                event_rel = event["timestamp"] - timestamps[0]
                plt.axvline(x=event_rel, color='red', linestyle='--', alpha=0.7)
                plt.text(event_rel, max(mem_usages)*0.95, f"t{i}", rotation=90, verticalalignment='top', color='red')
            plt.xlabel("Time (s)")
            plt.ylabel("Memory Usage (GB)")
            plt.title("Memory Usage Over Time")
            plt.xticks(tick_positions, tick_labels, rotation=45)
            plt.tight_layout()
            if save_dir:
                plt.savefig(os.path.join(save_dir, "memory_usage.png"))
                plt.close()
            else:
                plt.legend()
                plt.show()

        # --- Disk I/O Plot ---
        if self.disk_io:
            timestamps, read_rates, write_rates = zip(*self.disk_io, strict=False)
            rel_times = get_relative_times(self.disk_io)
            tick_positions, tick_labels = format_time_ticks(timestamps)
            plt.figure()
            plt.plot(rel_times, read_rates, label="Read Rate (MB/s)")
            plt.plot(rel_times, write_rates, label="Write Rate (MB/s)")
            # Annotate events
            for i, event in enumerate(self.events, start=1):
                event_rel = event["timestamp"] - timestamps[0]
                plt.axvline(x=event_rel, color='red', linestyle='--', alpha=0.7)
                plt.text(event_rel, max(max(read_rates), max(write_rates))*0.95, f"t{i}", rotation=90, verticalalignment='top', color='red')
            plt.xlabel("Time (s)")
            plt.ylabel("Disk I/O Rate (MB/s)")
            plt.title("Disk I/O Rates Over Time")
            plt.legend()
            plt.xticks(tick_positions, tick_labels, rotation=45)
            plt.tight_layout()
            if save_dir:
                plt.savefig(os.path.join(save_dir, "disk_io.png"))
                plt.close()
            else:
                plt.show()

        # --- GPU Usage Plot ---
        if self.gpu_available and any(usage_data for usage_data in self.gpu_usage):
            plt.figure(figsize=(12, 6))
            max_rel_time = 0
            for idx, usage_data in enumerate(self.gpu_usage):
                if usage_data:
                    timestamps, gpu_usages = zip(*usage_data, strict=False)
                    rel_times_gpu = [t - timestamps[0] for t in timestamps]
                    max_rel_time = max(max_rel_time, rel_times_gpu[-1])
                    plt.plot(rel_times_gpu, gpu_usages, label=f"{self.gpu_names[idx]}")
                    # Annotate events for this GPU plot
                    for i, event in enumerate(self.events, start=1):
                        event_rel = event["timestamp"] - timestamps[0]
                        plt.axvline(x=event_rel, color='red', linestyle='--', alpha=0.7)
                        plt.text(event_rel, max(gpu_usages)*0.95, f"t{i}", rotation=90, verticalalignment='top', color='red')
            num_ticks = min(10, len(rel_times_gpu))
            tick_positions = np.linspace(0, max_rel_time, num_ticks)
            tick_labels = [f"{t:.0f}s" for t in tick_positions]
            plt.xlabel("Time (s)")
            plt.ylabel("GPU Usage (%)")
            plt.ylim(0, 101)
            plt.title("GPU Usage Over Time")
            plt.grid(True)
            plt.legend()
            plt.xticks(tick_positions, tick_labels)
            plt.tight_layout()
            if save_dir:
                plt.savefig(os.path.join(save_dir, "gpu_usage.png"))
                plt.close()
            else:
                plt.show()

        # --- GPU Memory Usage Plot ---
        if self.gpu_available and any(memory_data for memory_data in self.gpu_memory_usage):
            plt.figure(figsize=(12, 6))
            max_rel_time = 0
            for idx, memory_data in enumerate(self.gpu_memory_usage):
                if memory_data:
                    timestamps, gpu_mem_usages = zip(*memory_data, strict=False)
                    rel_times_gpu = [t - timestamps[0] for t in timestamps]
                    max_rel_time = max(max_rel_time, rel_times_gpu[-1])
                    plt.plot(rel_times_gpu, gpu_mem_usages, label=f"{self.gpu_names[idx]}")
                    for i, event in enumerate(self.events, start=1):
                        event_rel = event["timestamp"] - timestamps[0]
                        plt.axvline(x=event_rel, color='red', linestyle='--', alpha=0.7)
                        plt.text(event_rel, max(gpu_mem_usages)*0.95, f"t{i}", rotation=90, verticalalignment='top', color='red')
            num_ticks = min(10, len(rel_times_gpu))
            tick_positions = np.linspace(0, max_rel_time, num_ticks)
            tick_labels = [f"{t:.0f}s" for t in tick_positions]
            plt.xlabel("Time (s)")
            plt.ylabel("GPU Memory Usage (GB)")
            plt.title("GPU Memory Usage Over Time")
            plt.grid(True)
            plt.legend()
            plt.xticks(tick_positions, tick_labels)
            plt.tight_layout()
            if save_dir:
                plt.savefig(os.path.join(save_dir, "gpu_memory_usage.png"))
                plt.close()
            else:
                plt.show()

        if save_dir:
            self.logger.info(f"Plots saved to {save_dir}")