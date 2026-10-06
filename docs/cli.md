# CLI and result schemas

The CLI is implemented with CLI11; unknown flags and malformed numeric values
are errors. Errors are one `error: ...` line on stderr and exit 2; deprecation
and assumption warnings are separate stderr lines. Commands perform no network
or GPU operations. Schema-v1 configs remain readable with deprecation warnings.

| Command | Options/inputs | Exit behavior |
| --- | --- | --- |
| budget | --config; --max-memory-gib or --max-memory-bytes; --max-concurrency; --format | 0 fit, 1 no fit, 2 error |
| simulate | --config; --fail-on-oom; --max-concurrency; --format | 0 ordinarily; 1 with fail flag and no fit; 2 error |
| schedule | --config; --events; --batch-tokens; --fail-on-preemption; --fail-on-truncation; --format | 1 infeasible or selected policy failure; otherwise 0; 2 error |
| sweep | --config; required --concurrency-min/max and --context-min/max; --batch-tokens; --max-points; --format | 0 completed grid; 2 invalid/error |
| compare | one or more config paths; --format | 0 valid comparison, 2 error |
| graph | --config | Mermaid output, 0 valid or 2 error |
| version | no arguments | CMake-generated version, exit 0 |
| completion | --shell bash | Bash completion definition, exit 0 |

All planning commands except graph accept `text`, `json`, `markdown`, `csv`;
default is text. Graph emits Mermaid directly. `--help` exits 0.
The memory override must be positive/finite and the two unit alternatives cannot
be combined. Nonnegative integer options reject signs, fractions and overflow.
`--max-concurrency` must be positive; a genuinely capped capacity sets
`max_concurrency_is_lower_bound=true` rather than pretending the cap is exact.
Sweep ranges are independent, inclusive, step one and checked before iteration;
`--max-points` defaults to 1000000. The batch-token override is independent of
the concurrency/context grid.

`--events` streams schema-2 event JSONL in binary mode (LF); existing files and
symlinks are refused. It never overwrites the input config. Output is not a
transactional file: an IO error can leave a partial event log. Events cannot
be replayed as real engine observations.

## JSON and formatting

Each JSON result has `kvbench_result: 2`, command name and `measurement: false`.
Budget/static output includes exact integer byte fields, weight-estimation flag,
whole blocks, capacity bounds, phase-paged memory, fit and risk. Schedule adds
completion status, step/peak counts, preemptions, actual recomputed tokens,
modeled latency/TTFT summaries and per-request outcomes. Missing times are JSON
null; zero-output requests have no TTFT. Static worst-case fields are separately
labelled, never merged with scheduled peak values.

Each event has `kvbench_event: 2`, step, modeled start/finish, running sequences,
allocated blocks and prefill/decode work. Identical config produces identical
serialized output on the same toolchain. There are no wall-clock timestamps.

Text quotes values to contain newlines. Markdown escapes table separators,
newlines and HTML characters. CSV quotes/doubles quotes in metric/value fields;
sweep CSV uses context_tokens,concurrent_requests,peak_bytes,fits,risk,
max_num_batched_tokens columns. Mermaid labels encode delimiters and controls.
`tests/cli_checks.py` runs the real binary and checks parsed results, exact
arrival sensitivity, exit codes, output safety and independent grid dimensions.

Illustrative Bash completion setup: `source <(kvbench completion --shell bash)`.
Import-HF and calibration commands are added in P7; they are not advertised as
implemented here.
