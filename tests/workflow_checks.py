"""Read workflow policy without triggering or changing GitHub state."""
import re
from pathlib import Path

root = Path(__file__).resolve().parents[1]
for path in (root / ".github/workflows").glob("*.yml"):
    text = path.read_text(encoding="utf-8")
    assert "permissions:" in text
    for action in re.findall(r"uses:\s*(\S+)", text):
        assert re.fullmatch(r"[^@]+@[0-9a-f]{40}", action), (path, action)
release = (root / ".github/workflows/release.yml").read_text(encoding="utf-8")
trigger = release.split("permissions:", 1)[0]
assert re.search(r"on:\n  push:\n    tags: \['v\*'\]\n$", trigger)
assert "pull_request" not in trigger and "branches:" not in trigger
assert "if: vars.KVBENCH_RELEASE_ENABLED == 'true'" in release
assert "environment: release" in release and "--expected-tag" in release
assert "contents: write" not in release
print("SHA pins, least permissions and tag-only opt-in/environment release checks passed")
