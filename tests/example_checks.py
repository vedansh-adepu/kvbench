"""Verify committed example assumptions using actual versioned CLI output."""
import json
import subprocess
import sys
from pathlib import Path

binary = sys.argv[1]
root = Path(__file__).resolve().parents[1] / "examples/v2"

def run(name, command="simulate"):
    result = subprocess.run([binary, command, "--config", str(root / f"{name}.json"), "--format", "json"], capture_output=True, text=True, check=True)
    return json.loads(result.stdout)

full = run("qwen2.5-72b-like")
rank = run("qwen2.5-72b-like-tp4")
assert full["budget"]["kv_data_bytes_per_token"] == 327680
assert rank["budget"]["kv_data_bytes_per_token"] == 81920
assert rank["budget"]["weights_bytes"] == (full["budget"]["weights_bytes"] + 3) // 4
assert rank["budget"]["activation_peak_bytes"] == full["budget"]["activation_peak_bytes"]
assert not full["static_worst_case"]["fits"] and rank["static_worst_case"]["fits"]
mla = run("deepseek-v3-like")
assert mla["budget"]["kv_data_bytes_per_token"] == 70272 and mla["budget"]["weights_bytes"] == 0
assert "cache-only" in mla["model_name"]
burst = run("bursty-arrivals", "schedule")["schedule"]
smooth = run("smoothed-arrivals", "schedule")["schedule"]
assert burst["peak_blocks"] == 40 and smooth["peak_blocks"] == 36
assert burst["preemptions"] == 3 and smooth["preemptions"] == 0
for path in sorted(root.glob("*.json")):
    result = subprocess.run([binary, "simulate", "--config", str(path), "--format", "json"], capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)["kvbench_result"] == 2
print("All v2 examples and tensor-parallel mapping checks passed")
