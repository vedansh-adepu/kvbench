# Comparing a real vLLM startup log

Calibration evidence is pending. The public discussion sample is used only to
test parser patterns and units; its model/config is unknown, so it is not a
kvbench accuracy result. No GPU measurement or calibration numbers are invented.

Capture startup output from one worker/GPU of the exact vLLM version and model
configuration you use. Preserve the model revision, device, tensor-parallel
layout, KV dtype, block size, utilization and runtime flags separately. Put
checkpoint weights and measured overheads in a schema-v2 config. kvbench models
one GPU pool; a multi-worker log must first be filtered to one worker/run.

Illustrative comparison (not run against a real engine here):

```sh
kvbench calibrate --vllm-log startup.log --config model.json --tolerance 5 --format json
```

The command compares absolute and percentage error per observed term. Percentage
uses the observed value as denominator. Missing terms stay null/not-compared;
missing CUDA graph memory is never invented as zero. A zero observed value with
nonzero prediction fails tolerance and reports null percent rather than infinity.
The log must contain KV pool bytes, GPU blocks or token capacity. Exit 0 means
within the selected tolerance, 1 outside, 2 unsupported/invalid input. It does
not automatically edit your config, verify provenance or prove engine fidelity.

## Supported patterns

The parser recognizes explicit GiB memory-profile fields for total GPU memory,
utilization bytes, weights, non-torch overhead, activation peak and remaining
KV cache. It also recognizes `Available KV cache memory: ... GiB`,
`CUDA graph memory takes ... GiB` / `CUDA graph memory: ... GiB`,
`# GPU blocks: N`, `GPU blocks: N`, `num_gpu_blocks=N` and
`GPU KV cache size: N tokens` (including correctly grouped thousands separators).
These patterns are not a promise of support for every engine/version's logs.
Conflicting repeated observations fail instead of silently choosing one worker.
Input is limited to 16 MiB and lines to 16384 bytes.

Historical `Loading model weights took ... GB` lines are ambiguous about binary
units and are ignored; use an explicit GiB memory-profile line or checkpoint size.
Printed memory values are rounded, so tiny percentage differences need not
represent engine error. Inspect units and choose a justified tolerance.

The parser-only fixture comes from
[vLLM discussion #13803](https://github.com/vllm-project/vllm/discussions/13803):
39.50 GiB at utilization 0.90, reported usable 35.55 GiB, weights 27.59 GiB,
non-torch 0.09 GiB, activation peak 1.48 GiB, KV pool 6.38 GiB. Those values
are observations from that public log, not predictions or a calibrated result.
`tests/test_integration.cpp` keeps public parsing separate from synthetic
comparison tests with a known toy config.

## Hugging Face import

`kvbench import-hf config.json --format json` returns a versioned result whose
`config` member is schema v2. Extract that member to create a planning config;
review `unmapped_fields` and warnings before using it. The importer never
fetches a model or checkpoint. It maps layer/head counts, head_dim or derived
hidden_size/heads, intermediate size, vocabulary, tied embeddings, expert counts,
sliding windows/layer types and MLA latent dimensions. Unknown attention types
and ambiguous expert mappings fail. A disabled sliding-window flag is honored.

The importer cannot infer measured weights/runtime overheads or all backend
layout details. MLA checkpoint weights must be supplied explicitly. Sliding MLA
without explicit layer geometry is rejected. The default dense weight estimate
assumes gated MLP and RMS norms; verify your architecture rather than assuming
an HF import proves exact parameter count or device support.
