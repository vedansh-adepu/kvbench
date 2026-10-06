#pragma once
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "kvbench/budget.hpp"

namespace kvbench {
enum class CompletionStatus : std::uint8_t {
  completed,
  truncated_time,
  truncated_steps,
  infeasible
};
struct RequestMetrics {
  std::string id;
  double arrival_ms = 0;
  std::optional<double> queueing_delay_ms;
  std::optional<double> ttft_ms;
  std::optional<double> latency_ms;
  Count output_tokens = 0;
  bool completed = false;
};
struct LatencySummary {
  double p50 = 0;
  double p90 = 0;
  double p99 = 0;
  double maximum = 0;
  Count samples = 0;
};
struct ScheduleStep {
  Count step = 0;
  double started_ms = 0;
  double finished_ms = 0;
  Count running = 0;
  Count allocated_blocks = 0;
  Count prefill_tokens = 0;
  Count decode_seqs = 0;
};
struct ScheduleResult {
  CompletionStatus status = CompletionStatus::completed;
  Count steps = 0;
  Count peak_blocks = 0;
  Count preemptions = 0;
  Count recomputed_tokens = 0;
  Count generated_tokens = 0;
  Count completed_requests = 0;
  double elapsed_ms = 0;
  double tokens_per_second = 0;
  std::optional<double> saturation_time_ms;
  LatencySummary ttft;
  LatencySummary latency;
  std::vector<RequestMetrics> requests;
};
using EventSink = std::function<void(const ScheduleStep&)>;
/// Deterministic modeled continuous batching. Uses the shared budget/block model;
/// decode-first, FCFS prefill, latest-admitted recompute preemption. Events are
/// sent immediately to the optional sink and never retained by the scheduler.
/// No wall-clock timing, network IO or GPU execution; times are uncalibrated
/// unless the caller has supplied calibrated coefficients. Bounds are explicit.
ScheduleResult schedule(const PlannerConfig& config, const EventSink& sink = {});
/// Stable completion spelling for versioned results.
std::string completion_name(CompletionStatus status);
}  // namespace kvbench
