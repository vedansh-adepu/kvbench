"""Measure planner CPU wall time and peak RSS, never GPU inference performance."""
from __future__ import annotations

import argparse
import json
import platform
import re
import statistics
import subprocess
import tempfile
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")
    binary = args.binary.resolve()
    base = {"schema_version": 2, "model": {"name": "CPU benchmark toy", "num_layers": 1, "num_attention_heads": 1, "num_kv_heads": 1, "head_dim": 1}, "weights": {"weights_bytes": 0}, "hardware": {"total_gpu_gib": 1}, "engine": {"gpu_memory_utilization": 1, "activation_peak_bytes": 0, "non_torch_overhead_bytes": 0, "cudagraph_memory_bytes": 0, "max_num_seqs": 256, "max_num_batched_tokens": 4096}, "workload": {"context_tokens": 32, "decode_tokens": 8}}
    output = ["# Planner CPU performance", "", "Measured CPU runtime/RSS of kvbench, not GPU inference throughput or calibration.", "", f"Host: {platform.platform()}, architecture {platform.machine()}.", f"Binary: {subprocess.check_output([str(binary), 'version'], text=True).strip()}, Release build."]
    if platform.system() == "Darwin":
        cpu = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        output.append(f"CPU: {cpu}.")
    output += ["", f"Command: `python3 bench/run.py --binary build-release/kvbench --runs {args.runs}`.", "", "Toy geometry isolates planner overhead: one layer/head/dimension, explicit zero", "weights/runtime allowances, 1 GiB pool; schedule prompts=32, outputs=8, 256", "sequence slots and 4096 batched tokens. It is not a real checkpoint model.", "Measured time includes process startup, JSON IO and formatting; stdout is discarded.", "", "| Workload | Median wall time (ms) | Peak child RSS (MiB) | Runs |", "| --- | ---: | ---: | ---: |"]
    with tempfile.TemporaryDirectory(prefix="kvbench-bench-") as directory:
        config = Path(directory) / "config.json"
        for name, command in [
            ("10,000-point independent sweep", ["sweep", "--concurrency-min", "1", "--concurrency-max", "100", "--context-min", "1", "--context-max", "100", "--format", "csv"]),
            ("10,000-request schedule", ["schedule", "--format", "json"]),
        ]:
            source = json.loads(json.dumps(base))
            if command[0] == "schedule":
                source["workload"]["requests"] = [{"id": f"r{index}", "arrival_ms": 0, "input_tokens": 32, "output_tokens": 8} for index in range(10000)]
            config.write_text(json.dumps(source), encoding="utf-8")
            # Validate the full run, not just subprocess success, before timing.
            if command[0] == "schedule":
                verified = subprocess.run([str(binary), *command, "--config", str(config)], capture_output=True, text=True, check=True)
                result = json.loads(verified.stdout)["schedule"]
                assert result["status"] == "completed" and result["completed_requests"] == 10000
            timings, peaks = [], []
            for _ in range(args.runs):
                invocation = [str(binary), *command, "--config", str(config)]
                if platform.system() == "Darwin":
                    invocation = ["/usr/bin/time", "-l", *invocation]
                elif platform.system() == "Linux":
                    invocation = ["/usr/bin/time", "-v", *invocation]
                started = time.perf_counter_ns()
                run = subprocess.run(invocation, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, check=True)
                timings.append((time.perf_counter_ns() - started) / 1e6)
                if platform.system() == "Darwin":
                    found = re.search(r"(\d+)\s+maximum resident set size", run.stderr)
                    if found:
                        peaks.append(int(found[1]) / 2**20)
                elif platform.system() == "Linux":
                    found = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", run.stderr)
                    if found:
                        peaks.append(int(found[1]) / 1024)
            peak = f"{max(peaks):.2f}" if peaks else "unavailable"
            output.append(f"| {name} | {statistics.median(timings):.2f} | {peak} | {args.runs} |")
    output += ["", "RSS is the maximum observed child-process RSS from the system time utility;", "no cross-machine performance or GPU provisioning accuracy is implied.", ""]
    print("\n".join(output), end="")


if __name__ == "__main__":
    main()
