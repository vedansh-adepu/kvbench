# Continuous-batching planning model

The scheduler in `src/scheduling.cpp` consumes schema-v2 request arrivals and
the same cache geometry/budget functions as static planning. It executes no
model kernels. All reported times are model outputs, not hardware measurements;
default coefficients are UNCALIBRATED.

Running requests keep admission order. Each step schedules decodes first,
one token per selected sequence within the batched-token limit. Crossing a
block boundary allocates another block. If space is unavailable, the latest
admitted running request is preempted: its blocks are freed and it returns to
the front of the waiting queue. Generated output progress is preserved, but
prompt and previous output context must be prefetched again. `recomputed_tokens`
counts actual recomputation performed, not hypothetical work left at truncation.

Existing prefills advance next; waiting requests are admitted FCFS when a
sequence slot and blocks are available. Chunked prefill can shorten a chunk to
fit the remaining token/block capacity; without chunking, the whole remaining
prefill must fit the step. A blocked earlier prefill is not bypassed by a later
waiting request. Arrivals tied in time use input order. At most `max_num_seqs`
are running and at most `max_num_batched_tokens` execute in a step.

Step duration is base_step_ms + prefill_tokens×prefill_ms_per_token +
decode_seqs×decode_ms_per_seq. Arrivals during a step become eligible at the
next step; there is no hidden 10 ms tick. When idle, time jumps directly to the
next arrival. TTFT is the modeled completion of the first decode step after
prefill; this simplification is not the timing of real engine sampling.
Queueing delay is first admission minus arrival; end-to-end latency is completion
minus arrival. Percentiles use nearest rank on the available samples.
Zero-output requests complete after prefill and have no TTFT sample.

Peak blocks measure logical pool occupancy, not GPU-resident VRAM: engines may
preallocate the whole pool at startup. They include final-token allocations before
completed requests release them. Saturation time is the first step start at which all pool blocks are held.
Throughput is committed generated tokens divided by elapsed modeled time from
zero, including idle arrival gaps; it is not measured GPU throughput.

Completion status is one of `completed`, `truncated_time`, `truncated_steps`,
`infeasible`. An individually unfit request is detected before stepping. A
configuration unable to make any progress is infeasible. Explicit `max_steps`
and `max_simulated_time_ms` bound work; a step crossing the time bound does not
commit output-token progress. All input requests remain represented in results,
including incomplete ones. Recompute thrashing may reach a configured bound;
a feasible memory footprint alone does not guarantee useful throughput.

Events go to an optional callback as they occur; the scheduler retains no event
history. Consumers can stream JSONL. Memory is proportional to request state and
active scheduling state, not the number of events. Tests may collect tiny event
histories to assert invariants. Callback IO failures propagate as errors.

The allocator uses aggregate all-layer block bundles. Sliding/hybrid layers
use capped per-layer byte demands, rounded up to that unit; backend-specific
hybrid cache groups, prefix sharing, speculative decoding and expanded MLA
layouts are outside this planning abstraction. See [budget](budget.md).

`tests/test_scheduling.cpp` verifies exact small fixtures, step invariants and
byte-identical serialized results across repeated runs. In the two-request,
three-block toy workload, simultaneous arrivals cause two preemptions and
p99 TTFT=5 modeled ms; spacing the second arrival at 10 ms yields zero
preemptions and p99 TTFT=2 modeled ms. These values are fixture assertions,
not calibration evidence.
