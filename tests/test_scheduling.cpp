#include "kvbench/scheduling.hpp"
#include "kvbench/json.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/catch_approx.hpp>

namespace {
kvbench::PlannerConfig fixture() {
  return kvbench::parse_planner_config(R"({"schema_version":2,"model":{"num_layers":1,"num_attention_heads":1,"num_kv_heads":1,"head_dim":1},"weights":{"weights_bytes":0},"hardware":{"total_gpu_bytes":12},"engine":{"gpu_memory_utilization":1,"non_torch_overhead_bytes":0,"activation_peak_bytes":0,"cudagraph_memory_bytes":0,"block_size":1,"max_num_seqs":2,"max_num_batched_tokens":2,"base_step_ms":1,"prefill_ms_per_token":0,"decode_ms_per_seq":0},"workload":{"context_tokens":1,"requests":[{"id":"first","input_tokens":1,"output_tokens":2},{"id":"second","input_tokens":1,"output_tokens":2}]}})");
}
kvbench::Json metrics_json(const kvbench::ScheduleResult& r) {
  kvbench::Json j = {{"status", kvbench::completion_name(r.status)}, {"steps", r.steps}, {"peak", r.peak_blocks}, {"preemptions", r.preemptions}, {"redo", r.recomputed_tokens}, {"elapsed", r.elapsed_ms}, {"generated", r.generated_tokens}, {"requests", kvbench::Json::array()}};
  for (const auto& m : r.requests) j["requests"].push_back({{"id", m.id}, {"output", m.output_tokens}, {"done", m.completed}, {"ttft", m.ttft_ms ? kvbench::Json(*m.ttft_ms) : kvbench::Json(nullptr)}, {"latency", m.latency_ms ? kvbench::Json(*m.latency_ms) : kvbench::Json(nullptr)}});
  return j;
}
}

TEST_CASE("B2 arrivals change preemption and modeled latency exactly", "[schedule][golden]") {
  auto c = fixture();
  const auto burst = kvbench::schedule(c);
  REQUIRE(burst.status == kvbench::CompletionStatus::completed);
  REQUIRE(burst.steps == 6);
  REQUIRE(burst.peak_blocks == 3);
  REQUIRE(burst.preemptions == 2);
  REQUIRE(burst.recomputed_tokens == 2);
  REQUIRE(burst.generated_tokens == 4);
  REQUIRE(burst.requests[0].ttft_ms == 2);
  REQUIRE(burst.requests[1].ttft_ms == 5);
  REQUIRE(burst.requests[0].latency_ms == 3);
  REQUIRE(burst.requests[1].latency_ms == 6);
  REQUIRE(burst.ttft.p99 == 5);
  c.workload.requests[1].arrival_ms = 10;
  const auto smooth = kvbench::schedule(c);
  REQUIRE(smooth.preemptions == 0);
  REQUIRE(smooth.recomputed_tokens == 0);
  REQUIRE(smooth.peak_blocks == 3);
  REQUIRE(smooth.elapsed_ms == 13);
  REQUIRE(smooth.ttft.p99 == 2);
  REQUIRE(smooth.completed_requests == 2);
  REQUIRE(smooth.tokens_per_second == Catch::Approx(4000.0 / 13));
}

TEST_CASE("B3 peak includes the final output allocation before release", "[schedule]") {
  auto c = fixture();
  c.workload.requests.resize(1);
  c.workload.requests[0].output_tokens = 1;
  c.total_gpu_bytes = 8;
  std::vector<kvbench::ScheduleStep> events;
  const auto result = kvbench::schedule(c, [&](const auto& step) { events.push_back(step); });
  REQUIRE(result.peak_blocks == 2);
  REQUIRE(events.size() == 2);
  REQUIRE(events.back().allocated_blocks == 2); // input+last output = 8 bytes.
  REQUIRE(result.completed_requests == 1);
  REQUIRE(result.requests[0].completed);
}

TEST_CASE("Chunked prefill respects both token and page budgets", "[schedule]") {
  auto c = fixture();
  c.workload.requests.resize(1);
  c.workload.requests[0].input_tokens = 5;
  c.workload.requests[0].output_tokens = 1;
  c.cache.block_size = c.engine.block_size = 2;
  c.total_gpu_bytes = 24;
  std::vector<kvbench::ScheduleStep> events;
  const auto result = kvbench::schedule(c, [&](const auto& step) { events.push_back(step); });
  REQUIRE(result.status == kvbench::CompletionStatus::completed);
  REQUIRE(result.steps == 4);
  REQUIRE(events[0].prefill_tokens == 2);
  REQUIRE(events[1].prefill_tokens == 2);
  REQUIRE(events[2].prefill_tokens == 1);
  REQUIRE(events[3].decode_seqs == 1);
  REQUIRE(result.peak_blocks == 3);
  c.engine.enable_chunked_prefill = false;
  REQUIRE(kvbench::schedule(c).status == kvbench::CompletionStatus::infeasible);
}

TEST_CASE("Running decodes consume token budget before FCFS prefills", "[schedule]") {
  auto c = fixture();
  c.total_gpu_bytes = 100;
  c.workload.requests[1].input_tokens = 4;
  c.workload.requests[1].output_tokens = 0;
  c.workload.requests[1].arrival_ms = 1;
  std::vector<kvbench::ScheduleStep> events;
  const auto r = kvbench::schedule(c, [&](const auto& step) { events.push_back(step); });
  REQUIRE(r.status == kvbench::CompletionStatus::completed);
  REQUIRE(events[1].decode_seqs == 1);
  REQUIRE(events[1].prefill_tokens == 1);
  REQUIRE(r.requests[0].ttft_ms == 2);
  REQUIRE_FALSE(r.requests[1].ttft_ms.has_value());
}

TEST_CASE("B10 idle jumps beyond 24 hours and explicit limits report status", "[schedule]") {
  auto c = fixture();
  c.workload.requests.resize(1);
  c.workload.requests[0].arrival_ms = 100000000;
  c.engine.max_simulated_time_ms = 100000100;
  REQUIRE(kvbench::schedule(c).elapsed_ms == 100000003);
  REQUIRE(kvbench::schedule(c).steps == 3);
  c.engine.max_simulated_time_ms = 86400000;
  auto r = kvbench::schedule(c);
  REQUIRE(r.status == kvbench::CompletionStatus::truncated_time);
  REQUIRE(r.elapsed_ms == 86400000);
  REQUIRE(r.steps == 0);
  c.workload.requests[0].arrival_ms = 0;
  c.engine.max_steps = 1;
  r = kvbench::schedule(c);
  REQUIRE(r.status == kvbench::CompletionStatus::truncated_steps);
  REQUIRE(r.completed_requests == 0);
  REQUIRE(r.steps == 1);
  c.engine.max_steps = 100;
  c.engine.max_simulated_time_ms = 0.5;
  r = kvbench::schedule(c);
  REQUIRE(r.status == kvbench::CompletionStatus::truncated_time);
  REQUIRE(r.generated_tokens == 0);
  REQUIRE(r.steps == 0);
}

TEST_CASE("Impossible single-request memory is reported without idle looping", "[schedule]") {
  auto c = fixture();
  c.total_gpu_bytes = 8; // Each request needs three blocks; only two exist.
  const auto r = kvbench::schedule(c);
  REQUIRE(r.status == kvbench::CompletionStatus::infeasible);
  REQUIRE(r.steps == 0);
  REQUIRE(r.requests.size() == 2);
  REQUIRE_FALSE(r.requests[0].completed);
}

TEST_CASE("Generated schedule invariants and determinism preserve every request", "[schedule][property]") {
  auto c = fixture();
  c.cache.block_size = c.engine.block_size = GENERATE(kvbench::Count{1}, kvbench::Count{2}, kvbench::Count{4});
  c.engine.max_num_seqs = GENERATE(kvbench::Count{1}, kvbench::Count{2});
  c.engine.max_num_batched_tokens = GENERATE(kvbench::Count{1}, kvbench::Count{2}, kvbench::Count{4});
  c.total_gpu_bytes = 16;
  const auto capacity = kvbench::estimate_budget(c).num_blocks;
  std::vector<kvbench::Json> events;
  const auto r = kvbench::schedule(c, [&](const auto& step) {
    REQUIRE(step.allocated_blocks <= capacity);
    REQUIRE(step.running <= c.engine.max_num_seqs);
    REQUIRE(step.prefill_tokens + step.decode_seqs <= c.engine.max_num_batched_tokens);
    REQUIRE(step.finished_ms > step.started_ms);
    events.push_back({{"step", step.step}, {"start", step.started_ms}, {"end", step.finished_ms}, {"blocks", step.allocated_blocks}});
  });
  REQUIRE(r.status == kvbench::CompletionStatus::completed);
  REQUIRE(r.completed_requests == 2);
  REQUIRE(r.generated_tokens == 4);
  REQUIRE(r.requests[0].id == "first");
  REQUIRE(r.requests[1].id == "second");
  for (const auto& request : r.requests) REQUIRE(request.completed);
  std::vector<kvbench::Json> again;
  const auto second = kvbench::schedule(c, [&](const auto& step) { again.push_back({{"step", step.step}, {"start", step.started_ms}, {"end", step.finished_ms}, {"blocks", step.allocated_blocks}}); });
  REQUIRE(events == again);
  REQUIRE(metrics_json(r).dump() == metrics_json(second).dump());
}

TEST_CASE("Zero-output requests complete at prefill and retain no fake TTFT", "[schedule]") {
  auto c = fixture();
  c.workload.requests.resize(1);
  c.workload.requests[0].output_tokens = 0;
  auto r = kvbench::schedule(c);
  REQUIRE(r.steps == 1);
  REQUIRE(r.generated_tokens == 0);
  REQUIRE(r.completed_requests == 1);
  REQUIRE(r.latency.samples == 1);
  REQUIRE(r.ttft.samples == 0);
  c.workload.requests.clear();
  r = kvbench::schedule(c);
  REQUIRE(r.status == kvbench::CompletionStatus::completed);
  REQUIRE(r.steps == 0);
  REQUIRE(r.tokens_per_second == 0);
}
