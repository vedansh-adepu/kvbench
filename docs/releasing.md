# Release preparation

The release-artifact workflow runs only for pushed `v*` tags, and only when
`KVBENCH_RELEASE_ENABLED` is the string `true`. It names the `release`
environment. Configure required reviewers and tag deployment restrictions in
GitHub repository settings before enabling the variable. YAML naming an
environment does not prove those protections exist.

Review all compiler-matrix, coverage, analyzer, sanitizer, fuzz and CodeQL jobs
first. Preserve actual engine calibration evidence separately; pending evidence
must stay explicit in release notes. This work branch has not been pushed,
tagged or released; no remote CI status badge is shown.

The workflow builds/tests Release on Linux/macOS/Windows, checks that the existing
tag equals the binary's generated version, and uploads platform/architecture
artifacts with SHA256 files. It creates no tag or GitHub Release. The maintainer
can review and attach those artifacts to a release separately. Binaries target
the runner OS/architecture, not every OS version or a universal macOS binary.
Windows builds use the static MSVC runtime; normal OS libraries remain required.

`scripts/package_binary.py` is tested on temporary output: wrong tags and existing
package names are refused, copied binaries report the same version, and checksum
bytes match exactly. A checksum detects corruption, not publisher authenticity.
Before distribution, inspect artifacts, verify checksums/version, and test them
on the intended target systems. Optional signing is a future task.
