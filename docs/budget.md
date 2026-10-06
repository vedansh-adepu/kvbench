# GPU budget and block allocation

`estimate_budget` validates its public configuration before computing:

`utilization_bytes = floor(total_gpu_bytes × gpu_memory_utilization)`

`kv_pool = utilization_bytes − weights − non_torch_overhead − activation_peak − cudagraph_memory`

All deductions are checked integer bytes. A fixed overhead deficit returns zero
KV pool and the deficit rather than unsigned wrap; an overflowing sum is an
error. Explicit weights are preferred; [estimated weights](static-model.md) are
labelled estimated. The three overhead defaults (0.25, 1 and 0.5 GiB respectively)
are UNCALIBRATED inputs. The activation allowance is fixed across phases; it is
not a kernel-memory trace or a hidden 0.25 scratch multiplier.

The pool capacity is floor((KV pool − constant scale metadata)/block bytes).
Block counts round down; per-sequence counts round up. A `num_blocks` override
may reduce capacity but cannot exceed the predicted memory budget. Remaining
bytes outside whole blocks are unavailable for sequence admission.

The breakdown includes total GPU bytes, utilization bytes, weights and whether
they are estimated, each overhead, KV pool, cache data/metadata sizes, block count,
maximum cached token slots, blocks per sequence and maximum concurrent sequences.
These are numeric model outputs, not measurements.

Full/MLA uniform models allocate bundles containing every layer's block.
For hybrid layers, sequence bytes sum independently capped layer allocations,
then round up to an aggregate bundle. This shared allocator abstraction prevents
fractional block admission. It does not reproduce backend-specific hybrid group
padding or physical tensor layout; calibrate a real engine and inspect its groups.

Fit/risk calculations use the effective whole-block capacity, including an
explicit block override, and the same rounded sequence bundles. Exact capacity
is `fits=true` / `at_capacity`. A reduced override cannot leave `risk=ok` while
reporting an allocation failure.

Tests in `tests/test_budget.cpp` check every deduction, missing capacity, scale
constants, override bounds, uint64 edges and the 4 GiB GQA golden pool. The toy
1000-byte GPU × 0.9 leaves 900 bytes; weights=100, non-torch=50, activation=100,
graphs=50 leaves a 600-byte pool. At 64 bytes/block, nine blocks fit; a 32-token
sequence takes two blocks, so four concurrent sequences fit.
