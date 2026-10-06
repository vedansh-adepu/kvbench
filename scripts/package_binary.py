"""Package one tested binary and its SHA256; never creates tags or releases."""
import argparse
import hashlib
import os
import shutil
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--binary", type=Path, required=True)
parser.add_argument("--platform", choices=("Linux", "macOS", "Windows"), required=True)
parser.add_argument("--arch", required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--expected-tag")
args = parser.parse_args()
version = subprocess.check_output([str(args.binary.resolve()), "version"], text=True).strip().split()
if len(version) != 2 or version[0] != "kvbench":
    raise SystemExit("unexpected binary version output")
if args.expected_tag and args.expected_tag != "v" + version[1]:
    raise SystemExit("tag does not match the tested binary version")
if not args.arch.isalnum():
    raise SystemExit("architecture label must be alphanumeric")
name = f"kvbench-{args.platform}-{args.arch}" + (".exe" if args.platform == "Windows" else "")
args.output.mkdir(parents=True, exist_ok=True)
target = args.output / name
checksum = args.output / (name + ".sha256")
if target.exists() or checksum.exists():
    raise SystemExit("package output already exists")
shutil.copyfile(args.binary, target)
if os.name != "nt":
    target.chmod(0o755)
digest = hashlib.sha256(target.read_bytes()).hexdigest()
checksum.write_text(f"{digest}  {name}\n", encoding="utf-8", newline="\n")
print(f"Packaged {name} ({version[1]}) with SHA256")
