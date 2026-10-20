#include <catch2/catch_test_macros.hpp>
#include <limits>
#include <cmath>

#include "kvbench/output.hpp"

#include "kvbench/budget.hpp"

namespace {
kvbench::PlannerConfig tiny() {
  return kvbench::parse_planner_config(
      R"({"schema_version":2,"model":{"num_layers":1,"num_attention_heads":1,"num_kv_heads":1,"head_dim":1,"hidden_size":1},"weights":{"weights_bytes":100},"hardware":{"total_gpu_bytes":1000},"engine":{"gpu_memory_utilization":0.9,"non_torch_overhead_bytes":50,"activation_peak_bytes":100,"cudagraph_memory_bytes":50,"block_size":16},"workload":{"context_tokens":17,"decode_tokens":15}})");
}
}  // namespace

TEST_CASE("Budget subtracts each resident/runtime term exactly once", "[budget]") {
  const auto b = kvbench::estimate_budget(tiny());
  REQUIRE(b.total_gpu_bytes == 1000);
  REQUIRE(b.utilization_bytes == 900);
  REQUIRE(b.weights.bytes == 100);
  REQUIRE_FALSE(b.weights.estimated);
  REQUIRE(b.non_torch_overhead_bytes == 50);
  REQUIRE(b.activation_peak_bytes == 100);
  REQUIRE(b.cudagraph_memory_bytes == 50);
  REQUIRE(b.kv_pool_bytes == 600);
  REQUIRE(b.cache.data_bytes_per_token == 4);
  REQUIRE(b.cache.bytes_per_block == 64);
  REQUIRE(b.num_blocks == 9);
  REQUIRE(b.max_tokens_in_cache == 144);
  REQUIRE(b.blocks_per_sequence == 2);
  REQUIRE(b.max_concurrent_sequences == 4);
  REQUIRE(b.workload.prefill_bytes == 228);
  REQUIRE(b.workload.decode_bytes == 228);
  REQUIRE(b.workload.fits);
  REQUIRE(b.overhead_deficit_bytes == 0);
}

TEST_CASE("Insufficient fixed overhead reports a deficit without wrapping", "[budget]") {
  auto c = tiny();
  c.weights.weights_bytes = 1000;
  const auto b = kvbench::estimate_budget(c);
  REQUIRE_FALSE(b.overheads_fit);
  REQUIRE(b.overhead_deficit_bytes == 300);
  REQUIRE(b.kv_pool_bytes == 0);
  REQUIRE(b.num_blocks == 0);
  REQUIRE_FALSE(b.workload.fits);
}

TEST_CASE("Block overrides cannot fabricate capacity", "[budget]") {
  auto c = tiny();
  c.engine.num_blocks = 4;
  REQUIRE(kvbench::estimate_budget(c).num_blocks == 4);
  REQUIRE(kvbench::estimate_budget(c).max_concurrent_sequences == 2);
  c.engine.num_blocks = 0;
  REQUIRE_FALSE(kvbench::estimate_budget(c).workload.fits);
  REQUIRE(kvbench::estimate_budget(c).workload.risk == "exceeded");
  c.engine.num_blocks = 10;
  REQUIRE_THROWS(kvbench::estimate_budget(c));
}

TEST_CASE("Constant scales reduce pool capacity once", "[budget]") {
  auto c = tiny();
  c.cache.quantization.dtype = kvbench::CacheDType::int8;
  c.cache.quantization.scale_mode = kvbench::ScaleMode::per_tensor;
  c.total_gpu_bytes = 100;
  c.engine.gpu_memory_utilization = 1;
  c.weights.weights_bytes = 0;
  c.engine.activation_peak_bytes = 0;
  c.engine.non_torch_overhead_bytes = 0;
  c.engine.cudagraph_memory_bytes = 0;
  const auto b = kvbench::estimate_budget(c);
  REQUIRE(b.cache.constant_metadata_bytes == 8);
  REQUIRE(b.cache.bytes_per_block == 32);
  REQUIRE(b.num_blocks == 2);  // floor((100-8)/32).
}

TEST_CASE("Budget supports representable maximal bytes without FP conversion UB", "[budget]") {
  auto c = tiny();
  c.total_gpu_bytes = std::numeric_limits<kvbench::Bytes>::max();
  c.engine.gpu_memory_utilization = 1;
  REQUIRE(kvbench::estimate_budget(c).utilization_bytes == c.total_gpu_bytes);
  c.weights.weights_bytes = c.total_gpu_bytes;
  REQUIRE_THROWS_AS(kvbench::estimate_budget(c), std::overflow_error);
}

TEST_CASE("Golden 4 GiB block pool matches exact GQA capacity", "[budget][golden]") {
  auto c = kvbench::parse_planner_config(
      R"({"schema_version":2,"model":{"num_layers":32,"num_attention_heads":32,"num_kv_heads":8,"head_dim":128},"weights":{"weights_bytes":0},"hardware":{"total_gpu_gib":4},"engine":{"gpu_memory_utilization":1,"non_torch_overhead_bytes":0,"activation_peak_bytes":0,"cudagraph_memory_bytes":0},"workload":{"context_tokens":2049}})");
  const auto b = kvbench::estimate_budget(c);
  REQUIRE(b.kv_pool_bytes == kvbench::gib_to_bytes(4));
  REQUIRE(b.num_blocks == 2048);
  REQUIRE(b.blocks_per_sequence == 129);
  REQUIRE(b.max_concurrent_sequences == 15);
  REQUIRE(b.max_tokens_in_cache == 32768);
}

TEST_CASE("vLLM mode preserves defaults and uses double ceil", "[budget][vllm]") {
  auto c = tiny();
  c.total_gpu_bytes = 1001;
  REQUIRE(kvbench::estimate_budget(c).utilization_bytes == 900);
  REQUIRE_FALSE(kvbench::budget_report(c, kvbench::estimate_budget(c), "budget")
                    .contains("vllm_capacity"));
  c.engine.vllm_compat = true;
  c.engine.max_model_len = 33;
  REQUIRE(kvbench::estimate_budget(c).utilization_bytes == 901);
  c.total_gpu_bytes = 1000;
  REQUIRE(kvbench::estimate_budget(c).utilization_bytes == 900);
}

TEST_CASE("Exact vLLM input bypasses unused overheads and reports fractional capacity",
          "[budget][vllm]") {
  auto c = tiny();
  c.engine.vllm_compat = true;
  c.engine.max_model_len = 33;
  c.engine.available_kv_pool_bytes = 128;
  c.weights.weights_bytes = std::numeric_limits<kvbench::Bytes>::max();
  c.engine.activation_peak_bytes = std::numeric_limits<kvbench::Bytes>::max();
  const auto b = kvbench::estimate_budget(c);
  REQUIRE(b.kv_pool_bytes == 128);
  REQUIRE(b.num_blocks == 2);
  REQUIRE(b.max_tokens_in_cache == 22);  // int((2 / ceil(33/16)) * 33).
  REQUIRE(std::abs(b.vllm_concurrency - 2.0 / 3) < 1e-12);
  REQUIRE(b.weights.bytes == 0);
  const auto j = kvbench::budget_report(c, b, "budget");
  REQUIRE_FALSE(j["vllm_capacity"].contains("log_line"));
  REQUIRE(j["vllm_capacity"]["log_lines"] == kvbench::Json::array({
          "GPU KV cache size: 22 tokens, Maximum concurrency for 33 tokens per request: 0.67x"}));
  c.engine.vllm_report_style = "split";
  const auto split = kvbench::budget_report(c, b, "budget");
  REQUIRE(split["vllm_capacity"]["log_lines"] == kvbench::Json::array({
          "GPU KV cache size: 22 tokens",
          "Maximum concurrency for 33 tokens per request: 0.67x"}));
  c.engine.vllm_report_style = "invalid";
  REQUIRE_THROWS(kvbench::estimate_budget(c));
  c.engine.vllm_report_style = "combined";
  c.engine.available_kv_pool_bytes = 0;
  REQUIRE(kvbench::estimate_budget(c).vllm_concurrency == 0);
  c.engine.vllm_compat = false;
  REQUIRE_THROWS(kvbench::estimate_budget(c));
}

TEST_CASE("Compatibility resolves replicated and sharded per-rank KV heads", "[budget][vllm]") {
  auto c = tiny();
  c.cache.model.num_attention_heads = 8;
  c.cache.model.num_kv_heads = 2;
  c.engine.vllm_compat = true;
  c.engine.max_model_len = 32;
  c.engine.available_kv_pool_bytes = 1024;
  c.engine.tensor_parallel_size = 8;
  REQUIRE(kvbench::estimate_budget(c).num_blocks == 16);  // One replicated KV head.
  c.engine.tensor_parallel_size = 1;
  REQUIRE(kvbench::estimate_budget(c).num_blocks == 8);
  c.cache.model.num_kv_heads = 6;
  c.engine.tensor_parallel_size = 2;
  REQUIRE(kvbench::estimate_budget(c).num_blocks == 5);  // Three KV heads per rank.
  c.engine.tensor_parallel_size = 4;
  REQUIRE_THROWS(kvbench::estimate_budget(c));
  c.engine.tensor_parallel_size = 1;
  c.cache.model.attention_type = kvbench::LayerKind::sliding;
  c.cache.model.sliding_window = 32;
  REQUIRE_THROWS(kvbench::estimate_budget(c));
}

TEST_CASE("Three published logs are conditional memory-rounding interval checks",
          "[budget][vllm][interval]") {
  // Issues 50705, 52802, 39010 in vllm-project/vllm. Assumed block size 16
  // and dense layout; config/backend provenance is not established by this test.
  struct Case {
    kvbench::Count layers, heads, kv_heads, dim, tp, length, tokens;
    long double gib;
    double concurrency;
    kvbench::CacheDType dtype;
  };
  for (const auto& row : {
      Case{24, 14, 2, 64, 1, 1024, 9668416, 55.32L, 9441.81, kvbench::CacheDType::fp8_e4m3},
      Case{28, 16, 8, 128, 1, 40960, 129744, 13.86L, 3.17, kvbench::CacheDType::bf16},
      Case{40, 32, 8, 128, 2, 128000, 311264, 11.87L, 2.43, kvbench::CacheDType::fp8_e4m3}}) {
    auto c = tiny();
    c.cache.model.num_layers = row.layers;
    c.cache.model.num_attention_heads = row.heads;
    c.cache.model.num_kv_heads = row.kv_heads;
    c.cache.model.head_dim = row.dim;
    c.cache.quantization.dtype = row.dtype;
    c.engine.vllm_compat = true;
    c.engine.tensor_parallel_size = row.tp;
    c.engine.max_model_len = row.length;
    c.engine.available_kv_pool_bytes = static_cast<kvbench::Bytes>(
        std::ceil((row.gib - 0.005L) * kvbench::gibibyte));
    const auto lo = kvbench::estimate_budget(c);
    c.engine.available_kv_pool_bytes = static_cast<kvbench::Bytes>(
        std::floor((row.gib + 0.005L) * kvbench::gibibyte));
    const auto hi = kvbench::estimate_budget(c);
    REQUIRE(lo.max_tokens_in_cache <= row.tokens);
    REQUIRE(row.tokens <= hi.max_tokens_in_cache);
    REQUIRE(lo.vllm_concurrency <= row.concurrency + 0.005);
    REQUIRE(row.concurrency - 0.005 <= hi.vllm_concurrency);
  }
}

TEST_CASE("Compatibility inputs remain strict and checked", "[budget][vllm]") {
  REQUIRE_THROWS(kvbench::parse_planner_config(
      R"({"schema_version":2,"model":{"num_layers":1,"num_attention_heads":1,"head_dim":1},"engine":{"vllm_compat":true,"max_model_len":1,"available_kv_pool_bytes":1.5}})"));
  auto c = tiny();
  c.engine.vllm_compat = true;
  REQUIRE_THROWS(kvbench::estimate_budget(c));
  c.engine.max_model_len = 1;
  REQUIRE_THROWS(kvbench::estimate_budget(c));
  c.workload.context_tokens = 1;
  c.workload.decode_tokens = 0;
  c.total_gpu_bytes = std::numeric_limits<kvbench::Bytes>::max();
  c.engine.gpu_memory_utilization = 1;
  REQUIRE_THROWS_AS(kvbench::estimate_budget(c), std::overflow_error);
  c.engine.available_kv_pool_bytes = 64;
  c.engine.max_model_len = (kvbench::Count{1} << 53) + 1;
  REQUIRE_THROWS_AS(kvbench::estimate_budget(c), std::overflow_error);
}
