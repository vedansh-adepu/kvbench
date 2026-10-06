"""Generate deterministic documentation from the actual kvbench binary."""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def generate(binary: Path) -> str:
    with tempfile.TemporaryDirectory(prefix="kvbench-results-") as directory:
        target = Path(directory) / "config.json"

        def run(config: dict, command: str = "simulate") -> dict:
            target.write_text(json.dumps(config), encoding="utf-8")
            result = subprocess.run([str(binary), command, "--config", str(target), "--format", "json"], capture_output=True, text=True, check=True)
            return json.loads(result.stdout)

        def example(name: str) -> dict:
            return json.loads((ROOT / "examples/v2" / f"{name}.json").read_text(encoding="utf-8"))

        rows = ["# Generated planning results", "", "Model output, not a hardware measurement; time model uncalibrated unless stated.", "", "Regenerate with `python3 scripts/gen_results.py --binary build-release/kvbench`.", "", "## Architecture cache geometry", "", "| Geometry | Data bytes/token |", "| --- | ---: |"]
        base = example("llama3-8b-like")
        mha = copy.deepcopy(base)
        mha["model"]["num_kv_heads"] = 32
        for label, config in [("MHA (32 KV heads)", mha), ("GQA (8 KV heads)", base), ("MLA latent (61 layers, 512+64)", example("deepseek-v3-like"))]:
            rows.append(f"| {label} | {run(config)['budget']['kv_data_bytes_per_token']:,} |")
        rows += ["", "MLA example excludes weights/runtime solely to isolate cache geometry.", "", "## 24 GB-class budget at different maximum lengths", "", "Assumed total GPU capacity is exactly 24,000,000,000 bytes, utilization=0.9;", "weights are estimated, runtime allowances are UNCALIBRATED. Maximum length", "includes all cached tokens, so decode_tokens=0 for this capacity comparison.", "", "| Cached length | Blocks/sequence | Pool blocks | Max sequences |", "| ---: | ---: | ---: | ---: |"]
        for context in (8192, 4096, 2048):
            c = copy.deepcopy(base)
            c["workload"] = {"context_tokens": context, "decode_tokens": 0, "concurrent_requests": 1}
            b = run(c)["budget"]
            rows.append(f"| {context} | {b['blocks_per_sequence']} | {b['num_blocks']} | {b['max_concurrent_sequences']} |")
        rows += ["", "## KV dtype capacity with explicit metadata assumptions", "", "Same estimated bf16 weights/runtime budget, cached length=4096. These", "metadata widths are planning assumptions; verify your exact backend/version.", "", "| KV dtype / metadata | Block data+metadata bytes | Constant metadata bytes | Pool blocks | Max sequences |", "| --- | ---: | ---: | ---: | ---: |"]
        for dtype, mode, zp in [("fp16", "none", 0), ("fp8_e4m3", "per_tensor", 0), ("int8", "per_token_head", 0), ("int4", "per_token_head", 2)]:
            c = copy.deepcopy(base)
            c["workload"] = {"context_tokens": 4096}
            c["engine"].update(kv_dtype=dtype, scale_mode=mode, scale_bytes=4, zero_point_bytes=zp)
            b = run(c)["budget"]
            rows.append(f"| {dtype} / {mode}, scale=4, zp={zp} | {b['bytes_per_block']:,} | {b['constant_metadata_bytes']} | {b['num_blocks']} | {b['max_concurrent_sequences']} |")
        rows += ["", "## Arrival sensitivity", "", "Isolated 40-block GQA fixture, identical requests. Weights/runtime are", "excluded; base=1 ms, prefill=0.01 ms/token, decode=0.1 ms/sequence.", "", "| Arrival pattern | Peak blocks | Preemptions | Recomputed tokens | p99 TTFT (modeled ms) | Status |", "| --- | ---: | ---: | ---: | ---: | --- |"]
        for label, name in [("Burst", "bursty-arrivals"), ("Smoothed", "smoothed-arrivals")]:
            s = run(example(name), "schedule")["schedule"]
            rows.append(f"| {label} | {s['peak_blocks']} | {s['preemptions']} | {s['recomputed_tokens']} | {s['ttft']['p99_ms']:.3f} | {s['status']} |")
        rows += ["", "## Independent sweep CSV excerpt", "", "The following is actual CLI CSV, not a hand-maintained approximation.", "", "```csv"]
        target.write_text(json.dumps(base), encoding="utf-8")
        sweep = subprocess.run([str(binary), "sweep", "--config", str(target), "--concurrency-min", "1", "--concurrency-max", "2", "--context-min", "2048", "--context-max", "2049", "--format", "csv"], capture_output=True, text=True, check=True)
        rows += sweep.stdout.rstrip().splitlines() + ["```", ""]
        return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    text = generate(args.binary.resolve())
    output = ROOT / "docs/results.md"
    if args.check:
        if not output.exists() or output.read_text(encoding="utf-8") != text:
            raise SystemExit("docs/results.md differs from current tool output; regenerate it")
        print("Generated results match current tool output")
    else:
        output.write_text(text, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
