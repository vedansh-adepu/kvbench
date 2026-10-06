# kvbench

kvbench is a C++ LLM inference capacity simulator that estimates KV-cache memory growth, batching pressure, quantization tradeoffs, decode-time load, and OOM risk from model and workload configs.

It is a planning tool, not an inference engine. It does not load model weights, run kernels, call model APIs, or require API keys.

## Features

- Simulate KV-cache growth for model and workload JSON configs
- Estimate prefill and decode memory pressure
- Compare KV quantization choices: `fp32`, `fp16`, `bf16`, `int8`, `int4`
- Check whether a workload fits a memory budget
- Sweep batch and concurrency ranges
- Compare multiple workloads
- Emit text, JSON, Markdown, CSV, and Mermaid outputs
- Run a deterministic request scheduler for arrival-based workloads

## Build

Requirements:

- CMake 3.20 or newer
- C++20 compiler

```sh
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

## Usage

```sh
./build/kvbench simulate --config examples/workloads/llama7b_chat.json
./build/kvbench simulate --config examples/workloads/llama7b_chat.json --format json
./build/kvbench simulate --config examples/workloads/llama7b_chat.json --format markdown
./build/kvbench compare examples/workloads/*.json
./build/kvbench budget --config examples/workloads/llama7b_chat.json --max-memory-gb 24
./build/kvbench sweep --config examples/workloads/llama7b_chat.json --batch-min 1 --batch-max 64
./build/kvbench sweep --config examples/workloads/llama7b_chat.json --batch-min 1 --batch-max 64 --format csv
./build/kvbench graph --config examples/workloads/llama7b_chat.json --format mermaid
./build/kvbench version
```

## Config format

```json
{
  "model": {
    "name": "llama-7b-like",
    "layers": 32,
    "attention_heads": 32,
    "kv_heads": 32,
    "head_dim": 128,
    "hidden_size": 4096,
    "dtype": "fp16"
  },
  "workload": {
    "context_tokens": 8192,
    "decode_tokens": 512,
    "concurrent_requests": 16,
    "batch_size": 16,
    "prefill_chunk_size": 1024
  },
  "system": {
    "gpu_memory_gb": 24,
    "reserved_memory_gb": 3,
    "kv_quantization": "fp16",
    "page_size_tokens": 16
  }
}
```

Scheduler workloads can provide request arrivals:

```json
{
  "model": { "name": "llama-7b-like", "layers": 32, "attention_heads": 32, "kv_heads": 32, "head_dim": 128, "hidden_size": 4096, "dtype": "fp16" },
  "system": { "gpu_memory_gb": 24, "reserved_memory_gb": 3, "kv_quantization": "fp16", "page_size_tokens": 16 },
  "requests": [
    {"id": "r1", "arrival_ms": 0, "input_tokens": 2048, "output_tokens": 256},
    {"id": "r2", "arrival_ms": 10, "input_tokens": 8192, "output_tokens": 512}
  ]
}
```

## Model assumptions

The formulas are documented in [docs/model.md](docs/model.md). Throughput pressure is a heuristic score for comparison only. It is not a hardware benchmark.

## License

MIT. See [LICENSE](LICENSE).
