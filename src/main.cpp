#include <CLI/CLI.hpp>
#include <cmath>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <optional>
#include <stdexcept>
#include "kvbench/output.hpp"
#include "kvbench/integration.hpp"
#include "kvbench/version.hpp"

namespace {
kvbench::PlannerConfig load(const std::string& path) {
  auto c = kvbench::load_planner_config(path);
  for (const auto& warning : c.warnings) std::cerr << "warning: " << warning << '\n';
  return c;
}
CLI::Validator unsigned_integer() {
  return CLI::Validator([](std::string& value) {
    return value.empty() || value.find_first_not_of("0123456789") != std::string::npos ? std::string("expected unsigned integer") : std::string{};
  }, "UINT64");
}
}
int main(int argc, char** argv) {
  CLI::App app{"LLM inference capacity planner: model output, not a hardware benchmark"};
  app.require_subcommand(1);
  app.footer("Examples:\n  kvbench budget --config model.json --format json\n  kvbench schedule --config arrivals.json --events events.jsonl\n  kvbench sweep --config model.json --concurrency-min 1 --concurrency-max 4 --context-min 2048 --context-max 4096 --format csv");
  std::string path, format = "text", events_path, shell = "bash";
  std::vector<std::string> paths;
  std::string hf_path, log_path;
  double tolerance = 5;
  std::optional<double> memory_gib;
  std::optional<kvbench::Bytes> memory_bytes;
  std::optional<kvbench::Count> max_concurrency, batch_tokens;
  kvbench::Count concurrency_min=1, concurrency_max=1, context_min=1, context_max=1, max_points=1000000;
  bool fail_oom=false, fail_preemption=false, fail_truncation=false;
  auto* budget = app.add_subcommand("budget", "Memory breakdown; exit 0 fit / 1 no fit / 2 error");
  auto* simulate = app.add_subcommand("simulate", "Static worst-case phase-paged memory");
  auto* scheduling = app.add_subcommand("schedule", "Continuous batching under a modeled time budget");
  auto* sweep = app.add_subcommand("sweep", "Independent concurrency and context ranges");
  auto* graph = app.add_subcommand("graph", "Escaped Mermaid memory-budget graph");
  for (auto* command : {budget,simulate,scheduling,sweep,graph}) command->add_option("--config",path)->required();
  for (auto* command : {budget,simulate,scheduling,sweep}) command->add_option("--format",format)->check(CLI::IsMember({"text","json","markdown","csv"}));
  for (auto* command : {budget,simulate}) command->add_option("--max-concurrency",max_concurrency)->check(unsigned_integer());
  budget->add_option("--max-memory-gib",memory_gib);
  budget->add_option("--max-memory-bytes",memory_bytes)->check(unsigned_integer());
  simulate->add_flag("--fail-on-oom",fail_oom);
  scheduling->add_flag("--fail-on-preemption",fail_preemption);
  scheduling->add_flag("--fail-on-truncation",fail_truncation);
  scheduling->add_option("--events",events_path,"Stream event JSONL to this file");
  scheduling->add_option("--batch-tokens",batch_tokens)->check(unsigned_integer());
  sweep->add_option("--concurrency-min",concurrency_min)->required()->check(unsigned_integer());
  sweep->add_option("--concurrency-max",concurrency_max)->required()->check(unsigned_integer());
  sweep->add_option("--context-min",context_min)->required()->check(unsigned_integer());
  sweep->add_option("--context-max",context_max)->required()->check(unsigned_integer());
  sweep->add_option("--batch-tokens",batch_tokens)->check(unsigned_integer());
  sweep->add_option("--max-points",max_points,"Explicit resource limit for grid output")->check(unsigned_integer());
  auto* import = app.add_subcommand("import-hf","Import architecture and report unmapped HF fields");
  import->add_option("config.json",hf_path)->required();
  import->add_option("--format",format)->check(CLI::IsMember({"text","json","markdown","csv"}));
  auto* calibrate = app.add_subcommand("calibrate","Compare predicted memory with a supplied startup log");
  calibrate->add_option("--config",path)->required();
  calibrate->add_option("--vllm-log",log_path)->required();
  calibrate->add_option("--tolerance",tolerance);
  calibrate->add_option("--format",format)->check(CLI::IsMember({"text","json","markdown","csv"}));
  auto* compare = app.add_subcommand("compare","Compare multiple config budgets");
  compare->add_option("configs",paths)->required();
  compare->add_option("--format",format)->check(CLI::IsMember({"text","json","markdown","csv"}));
  auto* completion = app.add_subcommand("completion","Print Bash completion definition");
  completion->add_option("--shell",shell)->check(CLI::IsMember({"bash"}));
  auto* version = app.add_subcommand("version","Print package version");
  try {
    app.parse(argc,argv);
    if (*version) { std::cout << "kvbench " << kvbench::version << '\n'; return 0; }
    if (*completion) { std::cout << "_kvbench_complete() { COMPREPLY=( $(compgen -W 'budget simulate schedule sweep compare graph import-hf calibrate version completion --help --config --format --max-memory-gib --max-memory-bytes --max-concurrency --events --batch-tokens --fail-on-oom --fail-on-preemption --fail-on-truncation --concurrency-min --concurrency-max --context-min --context-max --max-points' -- \"${COMP_WORDS[COMP_CWORD]}\") ); }\ncomplete -F _kvbench_complete kvbench\n"; return 0; }
    if (*import) { std::cout << kvbench::render_report(kvbench::import_hf(kvbench::read_text_input(hf_path)),format); return 0; }
    if (*calibrate) {
      const auto c=load(path);
      const auto report=kvbench::compare_calibration(c,kvbench::parse_vllm_log(kvbench::read_text_input(log_path)),tolerance);
      std::cout << kvbench::render_report(report,format);
      return report.at("within_tolerance").get<bool>() ? 0 : 1;
    }
    if (*compare) {
      kvbench::Json j={{"kvbench_result",2},{"command","compare"},{"measurement",false},{"configs",kvbench::Json::array()}};
      for (const auto& file : paths) { const auto c=load(file); j["configs"].push_back(kvbench::budget_report(c,kvbench::estimate_budget(c),"budget")); }
      std::cout << kvbench::render_report(j,format); return 0;
    }
    auto c=load(path);
    if (batch_tokens) c.engine.max_num_batched_tokens=*batch_tokens;
    if (memory_gib && memory_bytes) throw std::invalid_argument("choose one memory override");
    if (memory_gib) { if (!std::isfinite(*memory_gib) || *memory_gib<=0) throw std::invalid_argument("memory override must be finite and positive"); c.total_gpu_bytes=kvbench::gib_to_bytes(*memory_gib); }
    if (memory_bytes) c.total_gpu_bytes=*memory_bytes;
    kvbench::validate_planner_config(c);
    const auto b=kvbench::estimate_budget(c,max_concurrency);
    if (*graph) { std::cout << kvbench::budget_graph(c,b); return 0; }
    if (*budget || *simulate) {
      std::cout << kvbench::render_report(kvbench::budget_report(c,b,*budget ? "budget" : "simulate"),format);
      return !b.workload.fits && (*budget || fail_oom) ? 1 : 0;
    }
    if (*scheduling) {
      std::ofstream events;
      if (!events_path.empty()) {
        if (std::filesystem::exists(events_path) || std::filesystem::is_symlink(std::filesystem::symlink_status(events_path))) throw std::runtime_error("events path already exists; choose a fresh output path");
        events.open(events_path,std::ios::binary|std::ios::trunc);
        if (!events) throw std::runtime_error("cannot open events output");
      }
      const auto sink = [&](const kvbench::ScheduleStep& s) {
        if (events.is_open()) { events << kvbench::event_report(s).dump() << '\n'; if (!events) throw std::runtime_error("cannot write events output"); }
      };
      const auto r=kvbench::schedule(c,sink);
      if (events.is_open()) { events.flush(); if (!events) throw std::runtime_error("cannot flush events output"); }
      std::cout << kvbench::render_report(kvbench::schedule_report(c,b,r),format);
      return r.status==kvbench::CompletionStatus::infeasible || (fail_preemption && r.preemptions!=0) || (fail_truncation && r.status!=kvbench::CompletionStatus::completed) ? 1 : 0;
    }
    if (*sweep) {
      if (concurrency_min==0 || context_min==0 || concurrency_max<concurrency_min || context_max<context_min || max_points==0) throw std::invalid_argument("invalid sweep bounds");
      const auto points=kvbench::checked_mul(kvbench::checked_add(concurrency_max-concurrency_min,1),kvbench::checked_add(context_max-context_min,1));
      if (points>max_points) throw std::invalid_argument("sweep exceeds --max-points");
      kvbench::Json j={{"kvbench_result",2},{"command","sweep"},{"measurement",false},{"time_model_calibrated",c.engine.calibrated},{"rows",kvbench::Json::array()}};
      for (auto context=context_min;; context=kvbench::checked_add(context,1)) {
        for (auto concurrency=concurrency_min;; concurrency=kvbench::checked_add(concurrency,1)) {
          c.workload.context_tokens=context; c.workload.concurrent_requests=concurrency;
          const auto row=kvbench::estimate_budget(c);
          j["rows"].push_back({{"context_tokens",context},{"concurrent_requests",concurrency},{"peak_bytes",row.workload.peak_bytes},{"fits",row.workload.fits},{"risk",row.workload.risk},{"max_num_batched_tokens",c.engine.max_num_batched_tokens}});
          if (concurrency==concurrency_max) break;
        }
        if (context==context_max) break;
      }
      if (format=="csv") {
        std::cout << "context_tokens,concurrent_requests,peak_bytes,fits,risk,max_num_batched_tokens\n";
        for (const auto& row : j["rows"]) std::cout << row.at("context_tokens") << ',' << row.at("concurrent_requests") << ',' << row.at("peak_bytes") << ',' << row.at("fits") << ',' << row.at("risk").get<std::string>() << ',' << row.at("max_num_batched_tokens") << '\n';
      } else std::cout << kvbench::render_report(j,format);
      return 0;
    }
  } catch (const CLI::CallForHelp&) { std::cout << app.help(); return 0; }
  catch (const CLI::ParseError&) { std::cerr << "error: invalid command or option; see --help\n"; return 2; }
  catch (const std::exception& error) {
    const std::string message=error.what(); std::cerr << "error: " << message.substr(0,message.find_first_of("\r\n")) << '\n'; return 2;
  }
  return 2;
}
