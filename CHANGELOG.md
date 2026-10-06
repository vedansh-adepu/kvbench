# Changelog

## 1.0.0rc1 — Unreleased

### Breaking changes

- CMake minimum is 3.20; C++20, pinned FetchContent dependencies and presets.
- Schema v2 uses explicit byte/GiB units and strict unknown-key/type validation.
  Schema-less v1 files remain readable with warnings; old `_gb` meant GiB.
- Prototype C++ types/formulas removed. Use the checked planner interfaces.
- Coupled batch sweep flags replaced by independent context/concurrency ranges;
  budget override is `--max-memory-gib` or `--max-memory-bytes`.
- JSON results carry `kvbench_result: 2`; the uncalibrated pressure score is removed.

### Fixed

- B1: prefill pages its own context; the 978-byte fixture no longer falsely fits.
- B2: scheduled arrival results are separate from static worst-case outputs.
- B3: block allocation includes final output before release and explicit overheads.
- B4: checked uint64 arithmetic, safe conversions and checked sweep bounds.
- B5: integer fields never truncate fractions.
- B6: strict finite JSON and configuration values; arithmetic failures are errors.
- B7: CLI11 parses complete numeric tokens; budget overrides are validated.
- B8: wrong section types fail instead of silently defaulting.
- B9: validated UTF-8 and surrogate-pair decoding through nlohmann/json.
- B10: idle jumps and explicit time/step bounds with completion status; streamed events.
- B11: closed-form block capacity without an arbitrary concurrency cap.
- B12: exact equality is fits/at_capacity, including explicit block overrides.
- B13: Markdown, CSV, text and Mermaid contain escaped user labels.

### Modeling and quality

- D1: explicit/estimated resident weights; GQA shapes, gated MLP, tied heads, MoE.
- D2: separate explicit UNCALIBRATED runtime/activation/graph allowances.
- D3: measured pressure constants removed; modeled step-time coefficients are explicit.
- D4: documented CMake minimum matches preset/CTest commands.
- D5: independent sweep dimensions and separate batched-token override.
- D6: arrivals are admitted at modeled step boundaries; no hidden 10 ms tick.
- D7: internal bytes and correctly labelled GiB/decimal GB.
- D8: per-tensor/per-token-head/group metadata, zero points and int4 padding.
- D9: shared cache/block math for static and scheduled modes.
- D10: validated mutable public calculation configs and overrides.
- D11: configuration, math, scheduling, integration and formatting are separate modules.
- D12: exact/generative/CLI tests, coverage, bounded fuzzing and analyzer/CI gates.
- D13: CMake version generates the sole CLI version header.
- D14: removed prototype parser/formulas and unreachable scheduling rollback code.

### Added

- Latent MLA, per-layer sliding/hybrid geometry; documented allocator limitations.
- HF import and startup-log comparison with missing-term/provenance handling.
- Generated results checked against the binary; Release CPU benchmarks.
- SHA-pinned compiler matrix, analysis/coverage/fuzz/sanitizer workflows;
  gated release definitions, CodeQL, Scorecard and action dependency updates.

Real vLLM calibration and remote CI are pending. Local ASan runtime startup is
blocked; normal/UBSan, fuzzing, coverage and analysis results are documented.
