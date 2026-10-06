"""Exercise the actual CLI with temporary files; no services or GPUs."""
import csv
import io
import json
import subprocess
import sys
import tempfile
from pathlib import Path

binary = sys.argv[1]

def invoke(*args):
    return subprocess.run([binary, *map(str, args)], capture_output=True, text=True, check=False)

with tempfile.TemporaryDirectory(prefix="kvbench-cli-") as directory:
    root = Path(directory)
    config = root / "config.json"
    source = {"schema_version": 2, "model": {"name": 'a|b\n"quoted"', "num_layers": 1, "num_attention_heads": 1, "num_kv_heads": 1, "head_dim": 1}, "weights": {"weights_bytes": 0}, "hardware": {"total_gpu_bytes": 12}, "engine": {"gpu_memory_utilization": 1, "activation_peak_bytes": 0, "non_torch_overhead_bytes": 0, "cudagraph_memory_bytes": 0, "block_size": 1, "max_num_seqs": 2, "max_num_batched_tokens": 2, "base_step_ms": 1, "prefill_ms_per_token": 0, "decode_ms_per_seq": 0}, "workload": {"context_tokens": 1, "requests": [{"id": "first", "input_tokens": 1, "output_tokens": 2}, {"id": "second", "input_tokens": 1, "output_tokens": 2}]}}
    config.write_text(json.dumps(source), encoding="utf-8")
    for value in ("-1", "0", "inf", "nan", "24oops", "1e309"):
        r = invoke("budget", "--config", config, "--max-memory-gib", value)
        assert r.returncode == 2 and "error: " in r.stderr, (value, r)
        assert sum(line.startswith("error:") for line in r.stderr.splitlines()) == 1
    for args in (["--invented"], ["simulate", "--config", config, "--invented"], ["sweep", "--config", config, "--concurrency-min", "1.5", "--concurrency-max", "2", "--context-min", "1", "--context-max", "2"]):
        assert invoke(*args).returncode == 2
    assert invoke("--help").returncode == 0
    assert "Prepare" not in invoke("version").stdout
    r = invoke("budget", "--config", config, "--format", "json")
    assert r.returncode == 0
    report = json.loads(r.stdout)
    assert report["kvbench_result"] == 2 and report["budget"]["num_blocks"] == 3
    assert report["budget"]["weights_estimated"] is False
    assert invoke("budget", "--config", config, "--max-memory-bytes", "1").returncode == 1
    assert invoke("budget", "--config", config, "--max-memory-bytes", "-1").returncode == 2
    assert invoke("budget", "--config", config, "--max-concurrency", "0").returncode == 2
    assert json.loads(invoke("simulate", "--config", config, "--format", "json").stdout)["static_worst_case"]["fits"]
    assert invoke("simulate", "--config", config, "--fail-on-oom", "--format", "json").returncode == 0
    events = root / "events.jsonl"
    r = invoke("schedule", "--config", config, "--events", events, "--format", "json", "--fail-on-preemption")
    assert r.returncode == 1
    schedule = json.loads(r.stdout)["schedule"]
    assert schedule["preemptions"] == 2 and schedule["ttft"]["p99_ms"] == 5
    assert len(events.read_text(encoding="utf-8").splitlines()) == schedule["steps"]
    assert all(json.loads(line)["kvbench_event"] == 2 for line in events.read_text(encoding="utf-8").splitlines())
    assert invoke("schedule", "--config", config, "--events", config).returncode == 2
    assert json.loads(config.read_text(encoding="utf-8")) == source
    source["workload"]["requests"][1]["arrival_ms"] = 10
    config.write_text(json.dumps(source), encoding="utf-8")
    smooth = json.loads(invoke("schedule", "--config", config, "--format", "json").stdout)["schedule"]
    assert smooth["preemptions"] == 0 and smooth["ttft"]["p99_ms"] == 2
    first = invoke("schedule", "--config", config, "--format", "json").stdout
    assert first == invoke("schedule", "--config", config, "--format", "json").stdout
    grid = json.loads(invoke("sweep", "--config", config, "--concurrency-min", "1", "--concurrency-max", "2", "--context-min", "1", "--context-max", "2", "--batch-tokens", "3", "--format", "json").stdout)["rows"]
    assert [(row["context_tokens"], row["concurrent_requests"]) for row in grid] == [(1,1),(1,2),(2,1),(2,2)]
    assert all(row["max_num_batched_tokens"] == 3 for row in grid)
    assert invoke("sweep", "--config", config, "--concurrency-min", "0", "--concurrency-max", "2", "--context-min", "1", "--context-max", "2").returncode == 2
    assert invoke("sweep", "--config", config, "--concurrency-min", "1", "--concurrency-max", "2", "--context-min", "1", "--context-max", "2", "--max-points", "1").returncode == 2
    md = invoke("budget", "--config", config, "--format", "markdown").stdout
    assert 'a\\|b<br>"quoted"' in md
    rows = list(csv.reader(io.StringIO(invoke("budget", "--config", config, "--format", "csv").stdout)))
    assert ["model_name", source["model"]["name"]] in rows
    graph = invoke("graph", "--config", config).stdout
    assert 'a&#124;b&#10;&#34;quoted&#34;' in graph
    compare = json.loads(invoke("compare", config, config, "--format", "json").stdout)
    assert len(compare["configs"]) == 2
    assert "complete -F" in invoke("completion").stdout
    source["engine"]["max_steps"] = 1
    config.write_text(json.dumps(source), encoding="utf-8")
    assert invoke("schedule", "--config", config, "--fail-on-truncation").returncode == 1
    print("CLI numeric, schema, scheduler, output, sweep and safety regressions passed")
