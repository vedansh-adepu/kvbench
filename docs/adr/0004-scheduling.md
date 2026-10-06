# 0004 — Deterministic bounded scheduler

Schedule decodes first, then FCFS prefills. Preempt the latest admitted running
request and recompute its context. Time is a configurable model, and events are
streamed to an optional sink. Trade-off: this exposes workload effects without
running kernels, but sampled TTFT, hybrid grouping and backend timing are
simplifications. Explicit limits report truncation rather than silent horizons.
