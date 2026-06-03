# Estimation Model

kvbench is a capacity simulator. It does not execute an LLM and it does not claim exact hardware throughput.

The simulator estimates memory pressure from model shape, active tokens, request concurrency, KV quantization, page size, and reserved device memory.

## Dtypes

Explicit byte sizes are used:

| dtype | bytes |
|---|---:|
| fp32 | 4 |
| fp16 | 2 |
| bf16 | 2 |
| int8 | 1 |
| int4 | 0.5 |

## KV-cache bytes

Per-token KV bytes:

```text
layers * kv_heads * head_dim * 2 * kv_dtype_bytes
```

The factor of 2 accounts for keys and values.

Total KV bytes:

```text
per_token_kv_bytes * (context_tokens + decode_tokens) * concurrent_requests
```

Fragmentation is modeled by rounding each request's active token count up to `page_size_tokens` and charging the extra tokens as KV bytes.

## Memory budget

Usable memory:

```text
gpu_memory_gb - reserved_memory_gb
```

The simulator compares peak working memory against usable memory. Reserved memory is reported separately and included in total device pressure, but budget pass or fail uses usable memory.

## Prefill and decode pressure

Prefill pressure includes KV for input context plus a heuristic activation scratch term:

```text
layers * hidden_size * min(context_tokens, prefill_chunk_size) * active_batch * model_dtype_bytes * 0.25
```

Decode pressure includes final KV after generated tokens plus a one-step activation scratch term:

```text
layers * hidden_size * batch_size * model_dtype_bytes * 0.25
```

These scratch estimates are deliberately conservative planning signals. They are not kernel-level activation traces.

## OOM risk

OOM risk is based on `peak_working_memory / usable_memory`:

| ratio | risk |
|---:|---|
| `< 0.60` | low |
| `0.60-0.75` | moderate |
| `0.75-0.90` | elevated |
| `0.90-1.00` | high |
| `>= 1.00` | exceeded |

## Scheduler simulation

For configs with a `requests` array, kvbench runs a deterministic scheduler:

- requests become active at `arrival_ms`
- prefill advances by `prefill_chunk_size` tokens per step
- decode advances by one output token per step
- active KV tokens are tracked over time
- peak memory and OOM events are detected against usable memory

The scheduler is intentionally small and deterministic so output is reproducible in CI and local planning.
