# Security and input boundaries

kvbench reads untrusted JSON/config/log text and performs planning calculations.
It executes no model services, GPU kernels or downloaded Python/model code.
Inputs are bounded to 16 MiB; JSON nesting to 128; log lines to 16384 bytes.
Counts/dimensions and public calculation entry points validate ranges and reject
arithmetic overflow. HF import is local-file only and reports unmapped fields.

These limits and bounded fuzzing do not constitute a resource-exhaustion sandbox
or a proof of safety for every input. Do not run arbitrary downloaded binaries
or use estimates as a substitute for measuring your actual engine/device.

Event output refuses existing paths, but logs are not transactional or authenticated.
Do not include private startup data/model identifiers in public bug reports.
Dependencies are pinned by archive hashes; caller-provided offline source trees
are trusted and must be verified separately. CI runs with minimal job permissions.

Report a reproducible security issue privately through GitHub's security reporting
if enabled; otherwise contact the maintainer without publishing sensitive input.
See [quality](docs/quality.md) for the actual fuzzing/sanitizer scope and limitations.
