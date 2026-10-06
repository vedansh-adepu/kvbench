"""Execute the README quickstart and check its captured arrival-output excerpt."""
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
binary = str(Path(sys.argv[1]).resolve())
readme = (root / "README.md").read_text(encoding="utf-8")
block = re.search(r"<!-- tested: quickstart -->\s*```sh\n(.*?)```", readme, re.S).group(1)
for line in block.splitlines():
    args = shlex.split(line)
    args[0] = binary
    result = subprocess.run(args, cwd=root, capture_output=True, text=True, check=True)
    if "--format" in args:
        assert json.loads(result.stdout)["kvbench_result"] == 2
    else:
        assert result.stdout.startswith("kvbench ")
lines = ["pattern,peak_blocks,preemptions,recomputed_tokens,p99_ttft_ms,status"]
for label, example in [("burst", "bursty-arrivals"), ("smoothed", "smoothed-arrivals")]:
    result = subprocess.run([binary, "schedule", "--config", str(root / "examples/v2" / f"{example}.json"), "--format", "json"], capture_output=True, text=True, check=True)
    s = json.loads(result.stdout)["schedule"]
    lines.append(f"{label},{s['peak_blocks']},{s['preemptions']},{s['recomputed_tokens']},{s['ttft']['p99_ms']:.3f},{s['status']}")
expected = re.search(r"<!-- tested: arrival-output -->\s*```text\n(.*?)```", readme, re.S).group(1)
assert expected == "\n".join(lines) + "\n"
print("README quickstart and captured output match actual binary")
