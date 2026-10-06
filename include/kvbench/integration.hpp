#pragma once
#include <map>

#include "kvbench/output.hpp"

namespace kvbench {
struct LogObservations {
  std::map<std::string, Count> terms;
  std::optional<double> utilization;
};
/// Map a strict Hugging Face config into a validated schema-v2 config object;
/// report every unmapped root field. Does not infer checkpoint/calibration data.
Json import_hf(const std::string& source);
/// Parse bounded single-worker vLLM startup memory/capacity observations.
/// Explicit GiB is binary; ambiguous historical GB weight lines are ignored.
/// Repeated disagreeing observations are errors, never silently last-wins.
LogObservations parse_vllm_log(const std::string& source);
/// Compare model predictions with supplied observations, preserving missing terms.
/// Percentages use observed values; zero observed/nonzero predicted fails without
/// emitting infinity. This is a comparison, not verified calibration provenance.
Json compare_calibration(const PlannerConfig& config, const LogObservations& observed,
                         double tolerance_percent);
/// Read a bounded text input without writes; maximum 16 MiB.
std::string read_text_input(const std::string& path);
}  // namespace kvbench
