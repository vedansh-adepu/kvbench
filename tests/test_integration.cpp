#include "kvbench/integration.hpp"
#include <limits>
#include <catch2/catch_test_macros.hpp>

TEST_CASE("HF import maps GQA weights hints and reports every unmapped field", "[integration]") {
  const auto result=kvbench::import_hf(R"({"num_hidden_layers":32,"num_attention_heads":32,"num_key_value_heads":8,"hidden_size":4096,"intermediate_size":14336,"vocab_size":128256,"tie_word_embeddings":false,"head_dim":null,"model_type":"llama"})");
  const auto c=kvbench::parse_planner_config(result.at("config").dump());
  REQUIRE(c.cache.model.head_dim==128);
  REQUIRE(c.cache.model.num_kv_heads==8);
  REQUIRE(c.weights.intermediate_size==14336);
  REQUIRE_FALSE(c.weights.tie_word_embeddings);
  REQUIRE(result.at("unmapped_fields")==kvbench::Json::array({"head_dim","model_type"}));
  REQUIRE(result.at("measurement")==false);
  REQUIRE(kvbench::cache_layout(c.cache).data_bytes_per_token==131072);
}

TEST_CASE("HF import carries hybrid windows and latent MLA without invented weights", "[integration]") {
  auto result=kvbench::import_hf(R"({"num_hidden_layers":2,"num_attention_heads":4,"num_key_value_heads":1,"hidden_size":32,"sliding_window":8,"layer_types":["full_attention","sliding_attention"]})");
  auto c=kvbench::parse_planner_config(result.at("config").dump());
  REQUIRE(c.cache.model.layers[1].kind==kvbench::LayerKind::sliding);
  REQUIRE(c.cache.model.layers[1].window_tokens==8);
  result=kvbench::import_hf(R"({"num_hidden_layers":61,"num_attention_heads":128,"head_dim":128,"hidden_size":7168,"kv_lora_rank":512,"qk_rope_head_dim":64})");
  c=kvbench::parse_planner_config(result.at("config").dump());
  REQUIRE(c.cache.model.attention_type==kvbench::LayerKind::mla);
  REQUIRE(kvbench::cache_layout(c.cache).data_bytes_per_token==70272);
  REQUIRE_FALSE(c.weights.weights_bytes.has_value());
  REQUIRE_THROWS(kvbench::estimate_weights(c));
}

TEST_CASE("HF import honors disabled windows and rejects ambiguous mappings", "[integration]") {
  const auto result=kvbench::import_hf(R"({"num_hidden_layers":1,"num_attention_heads":1,"hidden_size":4,"sliding_window":8,"use_sliding_window":false})");
  const auto c=kvbench::parse_planner_config(result.at("config").dump());
  REQUIRE(c.cache.model.attention_type==kvbench::LayerKind::full);
  REQUIRE(result.at("unmapped_fields")==kvbench::Json::array({"sliding_window"}));
  REQUIRE_THROWS(kvbench::import_hf(R"({"num_hidden_layers":1,"num_attention_heads":1,"hidden_size":4,"layer_types":["mamba"]})"));
  REQUIRE_THROWS(kvbench::import_hf(R"({"num_hidden_layers":1,"num_attention_heads":1,"hidden_size":4,"num_experts":2,"num_local_experts":3})"));
  REQUIRE_THROWS(kvbench::import_hf(R"({"num_hidden_layers":8.9,"num_attention_heads":1,"hidden_size":4})"));
}

TEST_CASE("Public vLLM discussion sample tests parsing only not calibration", "[integration][public-sample]") {
  // Public machine-log sample: https://github.com/vllm-project/vllm/discussions/13803
  // Values reformatted below to isolate parser patterns. No model config is
  // known; NEVER compare this fixture as evidence of kvbench accuracy.
  const auto observed=kvbench::parse_vllm_log(
    "total_gpu_memory (39.50GiB) x gpu_memory_utilization (0.90) = 35.55GiB\n"
    "model weights take 27.59GiB; non_torch_memory takes 0.09GiB; "
    "activation peak memory takes 1.48GiB; reserved for KV Cache is 6.38GiB.\n");
  REQUIRE(observed.terms.at("total_gpu_bytes")==kvbench::gib_to_bytes(39.50));
  REQUIRE(observed.terms.at("utilization_bytes")==kvbench::gib_to_bytes(35.55));
  REQUIRE(observed.terms.at("weights_bytes")==kvbench::gib_to_bytes(27.59));
  REQUIRE(observed.terms.at("non_torch_overhead_bytes")==kvbench::gib_to_bytes(0.09));
  REQUIRE(observed.terms.at("activation_peak_bytes")==kvbench::gib_to_bytes(1.48));
  REQUIRE(observed.terms.at("kv_pool_bytes")==kvbench::gib_to_bytes(6.38));
  REQUIRE(observed.utilization==0.90);
  REQUIRE_FALSE(observed.terms.contains("cudagraph_memory_bytes"));
}

TEST_CASE("Startup parser rejects conflicts hostile values and size excess", "[integration]") {
  REQUIRE_THROWS(kvbench::parse_vllm_log("Available KV cache memory: 1GiB\nAvailable KV cache memory: 2GiB\n"));
  REQUIRE_THROWS(kvbench::parse_vllm_log("Available KV cache memory: 1e309GiB\n"));
  REQUIRE_THROWS(kvbench::parse_vllm_log("Available KV cache memory: -1GiB\n"));
  REQUIRE_THROWS(kvbench::parse_vllm_log("GPU KV cache size: 1,,2 tokens\n"));
  REQUIRE_THROWS(kvbench::parse_vllm_log("no known memory data"));
  REQUIRE_THROWS(kvbench::parse_vllm_log(std::string(16385,'x')));
  REQUIRE_THROWS(kvbench::parse_vllm_log(std::string(16*1024*1024+1,'x')));
  const auto observed=kvbench::parse_vllm_log("# GPU blocks: 2048, # CPU blocks: 0\nGPU KV cache size: 32,768 tokens\n");
  REQUIRE(observed.terms.at("num_blocks")==2048);
  REQUIRE(observed.terms.at("max_tokens_in_cache")==32768);
}

TEST_CASE("Calibration compares synthetic terms and preserves missing observations", "[integration]") {
  const auto c=kvbench::parse_planner_config(R"({"schema_version":2,"model":{"num_layers":1,"num_attention_heads":1,"head_dim":1},"weights":{"weights_bytes":0},"hardware":{"total_gpu_bytes":100},"engine":{"gpu_memory_utilization":1,"activation_peak_bytes":0,"non_torch_overhead_bytes":0,"cudagraph_memory_bytes":0,"block_size":1}})");
  kvbench::LogObservations observed;
  observed.terms={{"kv_pool_bytes",100},{"num_blocks",25}};
  const auto report=kvbench::compare_calibration(c,observed,0);
  REQUIRE(report.at("within_tolerance")==true);
  REQUIRE(report.at("measurement")==false);
  for (const auto& row:report.at("terms")) {
    if (row.at("term")=="weights_bytes") REQUIRE(row.at("observed").is_null());
  }
  observed.terms["kv_pool_bytes"]=50;
  REQUIRE(kvbench::compare_calibration(c,observed,99).at("within_tolerance")==false);
  REQUIRE(kvbench::compare_calibration(c,observed,100).at("within_tolerance")==true);
  observed.terms["kv_pool_bytes"]=0;
  const auto zero=kvbench::compare_calibration(c,observed,100);
  REQUIRE(zero.at("within_tolerance")==false);
  for (const auto& row:zero.at("terms")) if (row.at("term")=="kv_pool_bytes") REQUIRE(row.at("percent_error").is_null());
  REQUIRE_THROWS(kvbench::compare_calibration(c,observed,-1));
  REQUIRE_THROWS(kvbench::compare_calibration(c,observed,std::numeric_limits<double>::quiet_NaN()));
  observed.utilization=0;
  REQUIRE_THROWS(kvbench::compare_calibration(c,observed,5));
  observed.utilization.reset();
  observed.terms={{"weights_bytes",0}};
  REQUIRE_THROWS(kvbench::compare_calibration(c,observed,5));
}
