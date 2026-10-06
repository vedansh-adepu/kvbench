#include <exception>
#include <cmath>
#include <limits>
#include <sstream>
#include <iomanip>
#include <CLI/CLI.hpp>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "kvbench/config.hpp"
#include "kvbench/version.hpp"
#include "kvbench/estimator.hpp"

namespace {

constexpr auto kVersion = kvbench::version;

struct Args {
  std::vector<std::string> values;

  bool has(const std::string& flag) const {
    for (const auto& value : values) {
      if (value == flag) return true;
    }
    return false;
  }

  std::string get(const std::string& flag, const std::string& fallback = "") const {
    for (std::size_t i = 0; i + 1 < values.size(); ++i) {
      if (values[i] == flag) return values[i + 1];
    }
    return fallback;
  }
};

[[noreturn]] void usage_error(const std::string& message) {
  throw std::runtime_error(message + "\nRun 'kvbench help' for usage.");
}

int run_simulate(const Args& args) {
  const std::string path = args.get("--config");
  if (path.empty()) usage_error("simulate requires --config");
  const std::string format = args.get("--format", "text");
  const kvbench::Estimate estimate = kvbench::estimate(kvbench::load_config_file(path));
  if (format == "text") std::cout << kvbench::format_text_report(estimate);
  else if (format == "json") std::cout << kvbench::format_json_report(estimate);
  else if (format == "markdown") std::cout << kvbench::format_markdown_report(estimate);
  else usage_error("unsupported simulate format: " + format);
  return 0;
}

int run_compare(const Args& args) {
  std::vector<std::pair<std::string, kvbench::Estimate>> estimates;
  for (std::size_t i = 1; i < args.values.size(); ++i) {
    if (!args.values[i].empty() && args.values[i][0] == '-') continue;
    estimates.push_back({args.values[i], kvbench::estimate(kvbench::load_config_file(args.values[i]))});
  }
  if (estimates.empty()) usage_error("compare requires one or more config files");
  std::cout << kvbench::format_compare_table(estimates);
  return 0;
}

int run_budget(const Args& args) {
  const std::string path = args.get("--config");
  const std::string max_memory = args.get("--max-memory-gb");
  if (path.empty()) usage_error("budget requires --config");
  if (max_memory.empty()) usage_error("budget requires --max-memory-gb");

  kvbench::Config config = kvbench::load_config_file(path);
  config.system.gpu_memory_gb = std::stod(max_memory);
  kvbench::validate_config(config, "budget override");
  const kvbench::Estimate estimate = kvbench::estimate(config);
  std::cout << "budget: " << max_memory << " GiB total, "
            << kvbench::bytes_to_gb(estimate.usable_memory_bytes) << " GiB usable after reserved memory\n";
  std::cout << "peak working memory: " << kvbench::bytes_to_gb(estimate.peak_working_bytes) << " GiB\n";
  if (!estimate.fits_usable_memory) {
    std::cout << "failure: workload exceeds usable memory by "
              << kvbench::bytes_to_gb(-estimate.headroom_bytes) << " GiB\n";
    return 1;
  }
  std::cout << "ok: workload fits with " << kvbench::bytes_to_gb(estimate.headroom_bytes) << " GiB headroom\n";
  return 0;
}

int run_sweep(const Args& args) {
  const std::string path = args.get("--config");
  if (path.empty()) usage_error("sweep requires --config");
  const std::string min_text = args.get("--batch-min");
  const std::string max_text = args.get("--batch-max");
  if (min_text.empty() || max_text.empty()) usage_error("sweep requires --batch-min and --batch-max");
  const int batch_min = std::stoi(min_text);
  const int batch_max = std::stoi(max_text);
  if (batch_min <= 0 || batch_max < batch_min) usage_error("invalid batch sweep bounds");
  const std::string format = args.get("--format", "table");
  const auto rows = kvbench::sweep_batches(kvbench::load_config_file(path), batch_min, batch_max);
  if (format == "table") std::cout << kvbench::format_sweep_table(rows);
  else if (format == "csv") std::cout << kvbench::format_sweep_csv(rows);
  else usage_error("unsupported sweep format: " + format);
  return 0;
}

int run_graph(const Args& args) {
  const std::string path = args.get("--config");
  if (path.empty()) usage_error("graph requires --config");
  const std::string format = args.get("--format", "mermaid");
  if (format != "mermaid") usage_error("graph currently supports --format mermaid");
  std::cout << kvbench::format_mermaid_graph(kvbench::estimate(kvbench::load_config_file(path)));
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  CLI::App app{"Prototype kvbench CLI; v2 planner command migration follows core phases"};
  app.require_subcommand(1);
  std::string path;
  std::string format;
  std::vector<std::string> paths;
  double budget_memory = 0;
  int batch_min = 1;
  int batch_max = 1;
  auto* simulate = app.add_subcommand("simulate", "Static prototype estimate");
  simulate->add_option("--config", path)->required();
  simulate->add_option("--format", format)->check(CLI::IsMember({"text", "json", "markdown"}));
  auto* compare = app.add_subcommand("compare", "Compare prototype configs");
  compare->add_option("configs", paths)->required();
  auto* budget = app.add_subcommand("budget", "Check prototype memory fit");
  budget->add_option("--config", path)->required();
  budget->add_option("--max-memory-gb", budget_memory)->required();
  auto* sweep = app.add_subcommand("sweep", "Prototype coupled batch/concurrency sweep");
  sweep->add_option("--config", path)->required();
  sweep->add_option("--batch-min", batch_min)->required();
  sweep->add_option("--batch-max", batch_max)->required();
  sweep->add_option("--format", format)->check(CLI::IsMember({"table", "csv"}));
  auto* graph = app.add_subcommand("graph", "Prototype memory graph");
  graph->add_option("--config", path)->required();
  graph->add_option("--format", format)->check(CLI::IsMember({"mermaid"}));
  auto* version = app.add_subcommand("version", "Print version");
  auto* help = app.add_subcommand("help", "Print help");
  try {
    app.parse(argc, argv);
    Args args;
    if (*version) { std::cout << "kvbench " << kVersion << "\n"; return 0; }
    if (*help) { std::cout << app.help(); return 0; }
    if (*compare) { args.values = {"compare"}; args.values.insert(args.values.end(), paths.begin(), paths.end()); return run_compare(args); }
    args.values = {"command", "--config", path};
    if (!format.empty()) args.values.insert(args.values.end(), {"--format", format});
    if (*simulate) return run_simulate(args);
    if (*graph) return run_graph(args);
    if (*budget) {
      if (!std::isfinite(budget_memory) || budget_memory <= 0) throw std::invalid_argument("budget must be finite and positive");
      std::ostringstream exact_budget;
      exact_budget << std::setprecision(std::numeric_limits<double>::max_digits10) << budget_memory;
      args.values.insert(args.values.end(), {"--max-memory-gb", exact_budget.str()});
      return run_budget(args);
    }
    if (*sweep) {
      args.values.insert(args.values.end(), {"--batch-min", std::to_string(batch_min), "--batch-max", std::to_string(batch_max)});
      return run_sweep(args);
    }
  } catch (const CLI::CallForHelp&) {
    std::cout << app.help(); return 0;
  } catch (const CLI::ParseError&) {
    std::cerr << "error: invalid command or option; see --help\n"; return 2;
  } catch (const std::exception& error) {
    std::string message = error.what();
    const auto end = message.find_first_of("\r\n");
    std::cerr << "error: " << message.substr(0, end) << "\n"; return 2;
  }
  return 2;
}
