"""Read-only focused static analysis of project C++ sources."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--clang-tidy", default="clang-tidy")
    parser.add_argument("--cppcheck", default="cppcheck")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    extra = []
    if sys.platform == "darwin":
        sdk = subprocess.check_output(["xcrun", "--show-sdk-path"], text=True).strip()
        extra = [f"--extra-arg=-isysroot{sdk}", f"--extra-arg=-isystem{sdk}/usr/include/c++/v1"]
    for source in sorted((root / "src").glob("*.cpp")):
        subprocess.run([args.clang_tidy, str(source), "-p", str(args.build.resolve()), *extra], cwd=root, check=True)
    subprocess.run([args.cppcheck, "--enable=warning,style,performance,portability", "--check-level=exhaustive", "--error-exitcode=1", "--std=c++20", "--inline-suppr", "-I", "include", "src"], cwd=root, check=True)


if __name__ == "__main__":
    main()
