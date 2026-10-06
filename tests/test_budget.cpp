#include "kvbench/budget.hpp"
#include <limits>
#include <catch2/catch_test_macros.hpp>

namespace {
kvbench::PlannerConfig tiny() {
  return kvbench::parse_planner_config(R"({"schema_version":2,"model":{"num_layers":1,"num_attention_heads":1,"num_kv_heads":1,"head_dim":1,"hidden_size":1},"weights":{"weights_bytes":100},"hardware":{"total_gpu_bytes":1000},"engine":{"gpu_memory_utilization":0.9,"non_torch_overhead_bytes":50,"activation_peak_bytes":100,"cudagraph_memory_bytes":50,"block_size":16},"workload":{"context_tokens":17,"decode_tokens":15}})");
}
}

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
  REQUIRE(b.num_blocks == 2); // floor((100-8)/32).
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
  auto c = kvbench::parse_planner_config(R"({"schema_version":2,"model":{"num_layers":32,"num_attention_heads":32,"num_kv_heads":8,"head_dim":128},"weights":{"weights_bytes":0},"hardware":{"total_gpu_gib":4},"engine":{"gpu_memory_utilization":1,"non_torch_overhead_bytes":0,"activation_peak_bytes":0,"cudagraph_memory_bytes":0},"workload":{"context_tokens":2049}})");
  const auto b = kvbench::estimate_budget(c);
  REQUIRE(b.kv_pool_bytes == kvbench::gib_to_bytes(4));
  REQUIRE(b.num_blocks == 2048);
  REQUIRE(b.blocks_per_sequence == 129);
  REQUIRE(b.max_concurrent_sequences == 15);
  REQUIRE(b.max_tokens_in_cache == 32768);
}
