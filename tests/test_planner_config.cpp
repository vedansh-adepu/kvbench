#include "kvbench/planner_config.hpp"
#include "kvbench/json.hpp"

#include <limits>
#include "kvbench/config.hpp"
#include "kvbench/estimator.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

namespace {
kvbench::Json small_config() {
  return {{"schema_version", 2}, {"model", {{"num_layers", 2}, {"num_attention_heads", 4}, {"num_kv_heads", 2}, {"head_dim", 8}}}, {"weights", {{"weights_bytes", 0}}}, {"engine", {{"activation_peak_bytes", 0}, {"non_torch_overhead_bytes", 0}, {"cudagraph_memory_bytes", 0}}}, {"hardware", {{"total_gpu_gib", 4}}}, {"workload", {{"context_tokens", 17}, {"decode_tokens", 15}, {"concurrent_requests", 2}}}};
}
}

TEST_CASE("Strict JSON rejects duplicates nonfinite and invalid unicode", "[config]") {
  using namespace kvbench;
  REQUIRE(parse_json(R"({"label":"\ud83d\ude00"})").at("label").get<std::string>() == "\xf0\x9f\x98\x80"); // B9.
  REQUIRE(parse_json("18446744073709551615").get<Count>() == std::numeric_limits<Count>::max());
  const auto source = GENERATE(std::string{"{\"x\":1,\"x\":2}"}, std::string{"{\"nested\":{\"x\":1,\"x\":2}}"}, std::string{"1e309"}, std::string{"NaN"}, std::string{"[Infinity]"}, std::string{"\"\\ud800\""}, std::string{"\"\\udc00\""}, std::string{"{} trailing"}, std::string{"\"\xff\""});
  REQUIRE_THROWS_AS(parse_json(source), JsonError);
}

TEST_CASE("Strict JSON bounds size and recursion before unsafe processing", "[config]") {
  REQUIRE_THROWS_AS(kvbench::parse_json(std::string(16 * 1024 * 1024 + 1, ' ')), kvbench::JsonError);
  REQUIRE_THROWS_AS(kvbench::parse_json(std::string(20000, '[') + "0" + std::string(20000, ']')), kvbench::JsonError);
}

TEST_CASE("Schema v2 derives head dimensions and carries explicit bytes", "[config]") {
  using namespace kvbench;
  auto source = small_config();
  auto c = parse_planner_config(source.dump());
  REQUIRE(c.cache.model.hidden_size == 32);
  REQUIRE(c.cache.model.num_kv_heads == 2);
  REQUIRE(c.total_gpu_bytes == Bytes{4} * gibibyte);
  REQUIRE(c.weights.weights_bytes == 0);
  REQUIRE(c.engine.activation_peak_bytes == 0);
  REQUIRE(c.workload.context_tokens == 17);
  REQUIRE(c.workload.decode_tokens == 15);
  REQUIRE(c.warnings == std::vector<std::string>{"memory overheads and time model are UNCALIBRATED planning inputs"});
  source["model"].erase("head_dim");
  source["model"]["hidden_size"] = 32;
  REQUIRE(parse_planner_config(source.dump()).cache.model.head_dim == 8);
  source["model"]["hidden_size"] = 31;
  REQUIRE_THROWS(parse_planner_config(source.dump()));
}

TEST_CASE("V2 rejects unknown fields wrong section types and invalid integers", "[config]") {
  using namespace kvbench;
  auto source = small_config();
  const auto section = GENERATE("model", "weights", "engine", "hardware", "workload");
  source[section] = false; // B8: never silently default a mistyped section.
  REQUIRE_THROWS_AS(parse_planner_config(source.dump()), JsonError);
  source = small_config();
  source[section]["typo"] = 1;
  REQUIRE_THROWS_AS(parse_planner_config(source.dump()), JsonError);
  source = small_config();
  source["unexpected"] = true;
  REQUIRE_THROWS_AS(parse_planner_config(source.dump()), JsonError);
  for (const auto& value : {Json(8.9), Json(-0.5), Json(true), Json("8"), Json(nullptr), Json(-1), Json(0)}) {
    source = small_config();
    source["model"]["num_kv_heads"] = value;
    REQUIRE_THROWS(parse_planner_config(source.dump()));
  }
}

TEST_CASE("Schema and unit alternatives must be explicit and unambiguous", "[config]") {
  using namespace kvbench;
  auto source = small_config();
  const auto version = GENERATE(Json(1), Json(3), Json(2.0), Json("2"), Json(false));
  source["schema_version"] = version;
  REQUIRE_THROWS_AS(parse_planner_config(source.dump()), JsonError);
  source = small_config();
  source["hardware"]["total_gpu_bytes"] = 1024;
  REQUIRE_THROWS_AS(parse_planner_config(source.dump()), JsonError);
  source = small_config();
  source["engine"]["activation_peak_gib"] = 1;
  REQUIRE_THROWS_AS(parse_planner_config(source.dump()), JsonError);
}

TEST_CASE("V1 maps GiB units and old scratch with explicit deprecation", "[config]") {
  using namespace kvbench;
  auto c = parse_planner_config(R"({"model":{"name":"B1","layers":1,"attention_heads":1,"kv_heads":1,"head_dim":1,"hidden_size":100,"dtype":"fp16","typo":1},"workload":{"context_tokens":17,"decode_tokens":15,"concurrent_requests":1,"batch_size":1,"prefill_chunk_size":100},"system":{"gpu_memory_gb":1,"reserved_memory_gb":0,"page_size_tokens":16}})");
  REQUIRE(c.cache.model.head_dim == 1);
  REQUIRE(c.weights.weights_bytes == 0);
  REQUIRE(c.engine.gpu_memory_utilization == 1);
  REQUIRE(c.engine.activation_peak_bytes == 850); // 1*100*17*1*2/4; no hidden confidence claim.
  REQUIRE(c.cache.block_size == 16);
  REQUIRE(c.total_gpu_bytes == gibibyte);
  REQUIRE(c.warnings.size() == 4);
  REQUIRE(c.warnings[0].find("deprecated") != std::string::npos);
  REQUIRE(c.warnings[1] == "model: unknown field typo");
}

TEST_CASE("V2 hybrid MLA metadata and arrivals remain separate from static load", "[config]") {
  using namespace kvbench;
  auto source = small_config();
  source["model"]["layer_types"] = Json::array({"full", {{"type", "sliding"}, {"window_tokens", 8}}});
  source["engine"]["kv_dtype"] = "int4";
  source["engine"]["scale_mode"] = "group";
  source["engine"]["zero_point_bytes"] = 2;
  source["workload"]["requests"] = Json::array({{{"id", "first"}, {"arrival_ms", 0.25}, {"input_tokens", 128}, {"output_tokens", 16}}, {{"id", "second"}, {"arrival_ms", 10000}, {"input_tokens", 128}, {"output_tokens", 16}}});
  auto c = parse_planner_config(source.dump());
  REQUIRE(c.cache.model.layers[1].window_tokens == 8);
  REQUIRE(c.cache.quantization.zero_point_bytes == 2);
  REQUIRE(c.workload.context_tokens == 17); // Arrivals never rewrite the static worst case (B2).
  REQUIRE(c.workload.requests[0].arrival_ms == 0.25);
  REQUIRE(c.workload.requests[1].arrival_ms == 10000);
  source["workload"]["requests"][1]["id"] = "first";
  REQUIRE_THROWS(parse_planner_config(source.dump()));
}

TEST_CASE("Public config validation rejects mutations and count overflow", "[config]") {
  using namespace kvbench;
  const auto mutation = GENERATE(0, 1, 2, 3, 4, 5, 6, 7, 8);
  auto c = parse_planner_config(small_config().dump());
  switch (mutation) {
    case 0: c.engine.gpu_memory_utilization = std::numeric_limits<double>::quiet_NaN(); break;
    case 1: c.engine.block_size = 0; break;
    case 2: c.engine.max_num_seqs = 0; break;
    case 3: c.engine.preemption_mode = "swap"; break;
    case 4: c.engine.base_step_ms = -1; break;
    case 5: c.engine.max_simulated_time_ms = 0; break;
    case 6: c.workload.context_tokens = std::numeric_limits<Count>::max(); break;
    case 7: c.weights.num_experts = 0; break;
    case 8: c.schema_version = 1; break;
    default: FAIL("unknown mutation");
  }
  REQUIRE_THROWS(validate_planner_config(c));
}

TEST_CASE("Legacy compatibility rejects fractional integers and huge dimensions", "[config]") {
  using namespace kvbench;
  const auto value = GENERATE(Json(8.9), Json(-0.5), Json(2147483648ULL), Json(0), Json("8"));
  Json source = {{"model", {{"layers", 1}, {"attention_heads", 1}, {"head_dim", 1}, {"kv_heads", value}}}, {"workload", {{"context_tokens", 1}}}};
  REQUIRE_THROWS(parse_config_text(source.dump(), "test"));
  source["model"]["kv_heads"] = 1;
  source["system"] = false;
  REQUIRE_THROWS(parse_config_text(source.dump(), "test"));
  source["system"] = Json::object();
  source["model"]["attention_heads"] = 2147483647;
  source["model"]["head_dim"] = 2147483647;
  REQUIRE_THROWS(parse_config_text(source.dump(), "test"));
  source["model"]["attention_heads"] = 1;
  source["model"]["head_dim"] = 1;
  auto c = parse_config_text(source.dump(), "test");
  c.system.gpu_memory_gb = std::numeric_limits<double>::infinity();
  REQUIRE_THROWS(estimate(c));
}

TEST_CASE("Explicit zero dimensions are never interpreted as missing", "[config]") {
  auto source = small_config();
  source["model"]["hidden_size"] = 32;
  source["model"]["head_dim"] = 0;
  REQUIRE_THROWS(kvbench::parse_planner_config(source.dump()));
  source["model"]["head_dim"] = 8;
  source["model"]["hidden_size"] = 0;
  REQUIRE_THROWS(kvbench::parse_planner_config(source.dump()));
}

TEST_CASE("Unknown v1 sections are warned and ignored", "[config]") {
  auto c = kvbench::parse_planner_config(R"({"model":{"layers":1,"attention_heads":1,"head_dim":1},"workload":{"context_tokens":1},"engine":false,"weights":null,"hardware":"ignore"})");
  REQUIRE(c.total_gpu_bytes == kvbench::Bytes{24} * kvbench::gibibyte);
  REQUIRE(c.weights.weights_bytes == 0);
  REQUIRE(c.warnings.size() == 6);
}
