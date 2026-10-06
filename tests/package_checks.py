"""Exercise packaging in a temporary directory without creating release state."""
import hashlib
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

binary = Path(sys.argv[1]).resolve()
root = Path(__file__).resolve().parents[1]
os_name = {"Darwin": "macOS", "Windows": "Windows", "Linux": "Linux"}[platform.system()]
with tempfile.TemporaryDirectory(prefix="kvbench-package-") as directory:
    command = [sys.executable, str(root / "scripts/package_binary.py"), "--binary", str(binary), "--platform", os_name, "--arch", "test", "--output", directory]
    wrong = subprocess.run([*command, "--expected-tag", "vwrong"], capture_output=True, text=True, check=False)
    assert wrong.returncode != 0 and not list(Path(directory).iterdir())
    subprocess.run(command, capture_output=True, text=True, check=True)
    checksum = next(Path(directory).glob("*.sha256"))
    packaged = checksum.with_suffix("")
    assert checksum.read_text(encoding="utf-8") == hashlib.sha256(packaged.read_bytes()).hexdigest() + "  " + packaged.name + "\n"
    assert subprocess.check_output([str(packaged), "version"], text=True) == subprocess.check_output([str(binary), "version"], text=True)
    assert subprocess.run(command, capture_output=True, check=False).returncode != 0
print("Binary version, non-overwrite and checksum packaging checks passed")
