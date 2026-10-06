# Static worst case and resident weights

`estimate_static(config, available_bytes)` uses the shared checked cache model.
The supplied budget covers cache plus the fixed activation allowance; it excludes
weights, non-torch overhead and CUDA graphs. The budget phase derives that value.
This is a planning model, not an inference benchmark or a calibrated prediction.

Prefill pages the context C independently; decode pages C+D. For full attention,
per-sequence cache is ceil(C/P) × block bytes at prefill and ceil((C+D)/P) ×
block bytes at decode. Sliding layers use their own capped page counts from
[model](model.md). Per-tensor scales are charged once; activation allowance is
explicit and fixed, rather than scaling by a hidden scratch multiplier.

Maximum concurrency is floor((available − activation − constant metadata) /
max(prefill bytes per sequence, decode bytes per sequence)), or zero if fixed
terms do not fit. There is no arbitrary 99,999 cap. An explicit maximum bounds
the returned value only when true capacity exceeds it; that result must be
shown as a lower bound by the migrated CLI.

Fits means required ≤ available. Risk is `ok` below 80%, `high` from 80% to
below 95%, `critical` from 95% to below 100%, `at_capacity` at exactly 100%,
and `exceeded` above it. Threshold arithmetic does not multiply large byte
counts into an overflowing percentage.

The B1 regression has C=17, D=15, P=16, four KV bytes/token and an explicitly
mapped 850-byte activation allowance. Prefill needs 32 cached token slots
(128 bytes) plus 850 = 978 bytes, so it does not fit a 950-byte budget.
The fixture is tested in `tests/test_static_model.cpp`.

## Weight estimation assumptions

Explicit checkpoint bytes are preferred. The optional metadata allowance is
added to either explicit or estimated weights. For standard/GQA layers, the
estimator counts:

- Embeddings vocab×hidden, plus an untied head when configured; a final RMS norm.
- Q and O matrices hidden×(attention heads×head dimension), and K/V matrices
  hidden×(KV heads×head dimension), for every layer.
- Two or three MLP matrices hidden×intermediate per resident expert per layer.
- Two RMS norms per layer, and hidden×num_experts router matrices for MoE.

All configured experts, including shared experts, are resident; activated-expert
count is not used as a weight-memory discount. Each int4 tensor is independently
packed and rounded up. Biases, unusual projections, non-RMS norm parameters,
checkpoint padding and quantization metadata are not inferred.

MLA cache dimensions do not uniquely determine its attention projection shapes.
MLA weight estimation therefore requires explicit checkpoint weights rather than
inventing a dense-equivalent parameter count. This is a current limitation.

The toy regression has hidden=4, two layers, two attention heads, one KV head,
head dimension=2, intermediate=8 and vocabulary=10 with tied embeddings.
It has 348 estimated parameters and 696 bf16 bytes; changing to two resident
experts produces 556 parameters, including routers. These are tested hand
calculations, not real-model calibration results.
