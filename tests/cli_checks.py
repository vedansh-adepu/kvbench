"""Run the actual CLI offline; reject malformed options without tracebacks."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

binary = sys.argv[1]
with tempfile.TemporaryDirectory(prefix="kvbench-cli-") as directory:
    config = Path(directory) / "config.json"
    config.write_text(json.dumps({"model": {"layers": 1, "attention_heads": 1, "kv_heads": 1, "head_dim": 1, "hidden_size": 1}, "workload": {"context_tokens": 1}, "system": {"gpu_memory_gb": 24, "reserved_memory_gb": 3}}), encoding="utf-8")
    for value in ("-1", "3", "inf", "nan", "24oops", "1e309"):
        result = subprocess.run([binary, "budget", "--config", str(config), "--max-memory-gb", value], capture_output=True, text=True, check=False)
        assert result.returncode == 2, (value, result)
        assert result.stderr.startswith("error: ") and len(result.stderr.splitlines()) == 1
    for args in (["--invented"], ["simulate", "--config", str(config), "--invented"], ["sweep", "--config", str(config), "--batch-min", "1.5", "--batch-max", "2"]):
        assert subprocess.run([binary, *args], capture_output=True, check=False).returncode == 2
    assert subprocess.run([binary, "--help"], capture_output=True, check=False).returncode == 0
    print("CLI option/override regression checks passed")
