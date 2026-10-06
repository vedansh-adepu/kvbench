#pragma once
#include "kvbench/json.hpp"
#include "kvbench/scheduling.hpp"

namespace kvbench {
/// Versioned result schema 2; all memory fields are integer bytes.
Json budget_report(const PlannerConfig& config, const BudgetResult& budget,
                   const std::string& command);
/// Add modeled scheduler metrics and all request outcomes to schema-2 output.
Json schedule_report(const PlannerConfig& config, const BudgetResult& budget,
                     const ScheduleResult& result);
/// Serialize a single streamed event; no retained event history.
Json event_report(const ScheduleStep& step);
/// Render text/json/markdown/csv. Markdown and CSV escape all user labels.
std::string render_report(const Json& report, const std::string& format);
/// Escape Mermaid label delimiters/control characters without adding new lines.
std::string escape_mermaid(const std::string& label);
/// Mermaid budget graph; quoted user labels cannot inject graph statements.
std::string budget_graph(const PlannerConfig& config, const BudgetResult& budget);
}
