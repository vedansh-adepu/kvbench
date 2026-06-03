#include <cmath>
#include <exception>
#include <iostream>
#include <string>

#include "kvbench/config.hpp"
#include "kvbench/estimator.hpp"
#include "kvbench/scheduler.hpp"

namespace {

int failures = 0;

void check(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    ++failures;
  }
}

kvbench::Config sample_config(const std::string& quantization = "fp16") {
  const std::string text =
      "{"
      "\"model\":{\"name\":\"tiny\",\"layers\":2,\"attention_heads\":4,\"kv_heads\":4,\"head_dim\":8,\"hidden_size\":32,\"dtype\":\"fp16\"},"
      "\"workload\":{\"context_tokens\":128,\"decode_tokens\":16,\"concurrent_requests\":2,\"batch_size\":2,\"prefill_chunk_size\":64},"
      "\"system\":{\"gpu_memory_gb\":1,\"reserved_memory_gb\":0.1,\"kv_quantization\":\"" + quantization + "\",\"page_size_tokens\":16}"
      "}";
  return kvbench::parse_config_text(text, "inline");
}

void test_dtype_bytes() {
  check(kvbench::dtype_bytes(kvbench::parse_dtype("fp32")) == 4.0, "fp32 bytes");
  check(kvbench::dtype_bytes(kvbench::parse_dtype("fp16")) == 2.0, "fp16 bytes");
  check(kvbench::dtype_bytes(kvbench::parse_dtype("bf16")) == 2.0, "bf16 bytes");
  check(kvbench::dtype_bytes(kvbench::parse_dtype("int8")) == 1.0, "int8 bytes");
  check(kvbench::dtype_bytes(kvbench::parse_dtype("int4")) == 0.5, "int4 bytes");
}

void test_estimate_formula() {
  const kvbench::Config config = sample_config();
  const kvbench::Estimate estimate = kvbench::estimate(config);
  const double expected_per_token = 2.0 * 4.0 * 8.0 * 2.0 * 2.0;
  check(std::abs(estimate.per_token_kv_bytes - expected_per_token) < 0.001, "per-token KV formula");
  check(estimate.kv_cache_bytes > 0.0, "kv cache positive");
  check(estimate.prefill_bytes > 0.0, "prefill positive");
  check(estimate.decode_bytes > 0.0, "decode positive");
  check(estimate.recommended_max_concurrency > 0, "recommended concurrency positive");
}

void test_quantization_reduces_kv() {
  const kvbench::Estimate fp16 = kvbench::estimate(sample_config("fp16"));
  const kvbench::Estimate int8 = kvbench::estimate(sample_config("int8"));
  const kvbench::Estimate int4 = kvbench::estimate(sample_config("int4"));
  check(int8.kv_cache_bytes < fp16.kv_cache_bytes, "int8 reduces KV");
  check(int4.kv_cache_bytes < int8.kv_cache_bytes, "int4 reduces KV");
}

void test_scheduler() {
  const std::string text =
      "{"
      "\"model\":{\"name\":\"scheduled\",\"layers\":2,\"attention_heads\":4,\"kv_heads\":4,\"head_dim\":8,\"hidden_size\":32,\"dtype\":\"fp16\"},"
      "\"workload\":{\"context_tokens\":1,\"decode_tokens\":1,\"concurrent_requests\":1,\"batch_size\":1,\"prefill_chunk_size\":64},"
      "\"system\":{\"gpu_memory_gb\":1,\"reserved_memory_gb\":0.1,\"kv_quantization\":\"fp16\",\"page_size_tokens\":16},"
      "\"requests\":["
      "{\"id\":\"r1\",\"arrival_ms\":0,\"input_tokens\":128,\"output_tokens\":4},"
      "{\"id\":\"r2\",\"arrival_ms\":10,\"input_tokens\":64,\"output_tokens\":2}"
      "]"
      "}";
  const kvbench::Config config = kvbench::parse_config_text(text, "scheduler");
  const kvbench::SchedulerResult result = kvbench::simulate_scheduler(config);
  check(!result.events.empty(), "scheduler events produced");
  check(result.peak_memory_bytes > 0.0, "scheduler peak memory positive");
  check(!result.oom, "scheduler should fit tiny config");
}

void test_format_outputs() {
  const kvbench::Estimate estimate = kvbench::estimate(sample_config());
  check(kvbench::format_json_report(estimate).find("\"model_name\"") != std::string::npos, "json report");
  check(kvbench::format_markdown_report(estimate).find("| Metric | Value |") != std::string::npos, "markdown report");
  check(kvbench::format_mermaid_graph(estimate).find("xychart-beta") != std::string::npos, "mermaid graph");
  check(kvbench::format_sweep_csv(kvbench::sweep_batches(sample_config(), 1, 3)).find("batch,peak_gb") == 0, "sweep csv");
}

}  // namespace

int main() {
  try {
    test_dtype_bytes();
    test_estimate_formula();
    test_quantization_reduces_kv();
    test_scheduler();
    test_format_outputs();
  } catch (const std::exception& error) {
    std::cerr << "unhandled exception: " << error.what() << "\n";
    return 1;
  }
  if (failures != 0) {
    std::cerr << failures << " test failure(s)\n";
    return 1;
  }
  std::cout << "all tests passed\n";
  return 0;
}
