#include "kvbench/static_model.hpp"
#include "kvbench/json.hpp"
#include <limits>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

namespace {
kvbench::PlannerConfig toy() {
  return kvbench::parse_planner_config(R"({"schema_version":2,"model":{"num_layers":2,"num_attention_heads":2,"num_kv_heads":1,"head_dim":2,"hidden_size":4},"weights":{"intermediate_size":8,"vocab_size":10},"engine":{"activation_peak_bytes":0,"block_size":1},"workload":{"context_tokens":1}})");
}
}

TEST_CASE("B1 prefill pages context independently of final tokens", "[static][golden]") {
  auto c = kvbench::parse_planner_config(R"({"model":{"layers":1,"attention_heads":1,"kv_heads":1,"head_dim":1,"hidden_size":100,"dtype":"fp16"},"workload":{"context_tokens":17,"decode_tokens":15,"concurrent_requests":1,"batch_size":1,"prefill_chunk_size":100},"system":{"gpu_memory_gb":1,"reserved_memory_gb":0,"page_size_tokens":16}})");
  const auto result = kvbench::estimate_static(c, 950);
  REQUIRE(result.prefill_cache_bytes == 128); // ceil(17/16)*16*4, not reused 32-token padding.
  REQUIRE(result.prefill_bytes == 978); // 128 + the explicit mapped 850-byte activation allowance.
  REQUIRE(result.decode_cache_bytes == 128);
  REQUIRE(result.peak_bytes == 978);
  REQUIRE_FALSE(result.fits);
  REQUIRE(result.risk == "exceeded");
}

TEST_CASE("B11 concurrency has no artificial 99999 ceiling", "[static]") {
  auto c = toy();
  const auto bytes = kvbench::sequence_cache_bytes(c.cache, 1);
  REQUIRE(kvbench::estimate_static(c, bytes * 100000).max_concurrency == 100000);
  REQUIRE(kvbench::estimate_static(c, bytes * 100001).max_concurrency == 100001);
  auto e = kvbench::estimate_static(c, bytes * 100001, 100000);
  REQUIRE(e.max_concurrency == 100000);
  REQUIRE(e.concurrency_bounded);
  REQUIRE_FALSE(kvbench::estimate_static(c, bytes * 100000, 100000).concurrency_bounded);
  REQUIRE_THROWS(kvbench::estimate_static(c, bytes, 0));
}

TEST_CASE("B12 fit and risk agree at exact byte thresholds", "[static]") {
  using kvbench::memory_risk;
  REQUIRE(memory_risk(79, 100) == "ok");
  REQUIRE(memory_risk(80, 100) == "high");
  REQUIRE(memory_risk(94, 100) == "high");
  REQUIRE(memory_risk(95, 100) == "critical");
  REQUIRE(memory_risk(99, 100) == "critical");
  REQUIRE(memory_risk(100, 100) == "at_capacity");
  REQUIRE(memory_risk(101, 100) == "exceeded");
  REQUIRE(memory_risk(0, 0) == "at_capacity");
  auto c = toy();
  auto e = kvbench::estimate_static(c, kvbench::sequence_cache_bytes(c.cache, 1));
  REQUIRE(e.fits);
  REQUIRE(e.risk == "at_capacity");
  REQUIRE(memory_risk(std::numeric_limits<kvbench::Bytes>::max()-1, std::numeric_limits<kvbench::Bytes>::max()) == "critical");
}

TEST_CASE("Resident weight estimator matches explicit dense and MoE hand math", "[weights][golden]") {
  auto c = toy();
  // Embedding=40, final norm=4; per layer Q/O=32,K/V=16,MLP=96,norms=8.
  auto e = kvbench::estimate_weights(c);
  REQUIRE(e.parameters == 348);
  REQUIRE(e.bytes == 696);
  REQUIRE(e.estimated);
  c.weights.tie_word_embeddings = false;
  REQUIRE(kvbench::estimate_weights(c).parameters == 388);
  c.weights.tie_word_embeddings = true;
  c.weights.num_experts = 2;
  // Every expert resident: per layer 48 attention+192 MLP+8 norms+8 router.
  REQUIRE(kvbench::estimate_weights(c).parameters == 556);
  c.weights.num_shared_experts = 1;
  REQUIRE(kvbench::estimate_weights(c).parameters == 748);
  c.weights.num_experts = 1;
  c.weights.num_shared_experts = 0;
  c.weights.gated_mlp = false;
  REQUIRE(kvbench::estimate_weights(c).parameters == 284);
  c.weights.weights_bytes = 123;
  c.weights.quantization_overhead_bytes = 7;
  e = kvbench::estimate_weights(c);
  REQUIRE(e.bytes == 130);
  REQUIRE_FALSE(e.estimated);
}

TEST_CASE("Weight int4 packing rounds each tensor not just the final total", "[weights]") {
  auto c = toy();
  c.cache.model.hidden_size = 1;
  c.cache.model.num_attention_heads = 1;
  c.cache.model.head_dim = 1;
  c.cache.model.num_layers = 1;
  c.weights.intermediate_size = 1;
  c.weights.vocab_size = 1;
  c.weights.dtype = kvbench::CacheDType::int4;
  // 11 independent one-element tensors each occupy one byte, not ceil(11/2).
  REQUIRE(kvbench::estimate_weights(c).parameters == 11);
  REQUIRE(kvbench::estimate_weights(c).bytes == 11);
  c.cache.model.attention_type = kvbench::LayerKind::mla;
  c.cache.model.kv_lora_rank = 1;
  c.cache.model.qk_rope_head_dim = 1;
  REQUIRE_THROWS(kvbench::estimate_weights(c));
  c.weights.weights_bytes = 10;
  REQUIRE(kvbench::estimate_weights(c).bytes == 10);
}

TEST_CASE("Generated static capacity boundary fits and next sequence fails", "[static][property]") {
  auto c = toy();
  c.cache.block_size = c.engine.block_size = GENERATE(kvbench::Count{1}, kvbench::Count{16});
  c.workload.context_tokens = GENERATE(kvbench::Count{1}, kvbench::Count{17}, kvbench::Count{129});
  c.workload.decode_tokens = GENERATE(kvbench::Count{0}, kvbench::Count{15});
  const auto available = kvbench::Bytes{100000};
  const auto max = kvbench::estimate_static(c, available).max_concurrency;
  c.workload.concurrent_requests = max;
  REQUIRE(kvbench::estimate_static(c, available).fits);
  c.workload.concurrent_requests = max + 1;
  REQUIRE_FALSE(kvbench::estimate_static(c, available).fits);
}
