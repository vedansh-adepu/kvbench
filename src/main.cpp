#include <exception>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "kvbench/config.hpp"
#include "kvbench/estimator.hpp"

namespace {

constexpr const char* kVersion = "0.1.0";

struct Args {
  std::vector<std::string> values;

  explicit Args(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) values.emplace_back(argv[i]);
  }

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

void print_usage() {
  std::cout
      << "kvbench " << kVersion << "\n\n"
      << "Usage:\n"
      << "  kvbench simulate --config <file> [--format text|json|markdown]\n"
      << "  kvbench compare <config> [config...]\n"
      << "  kvbench budget --config <file> --max-memory-gb <gb>\n"
      << "  kvbench sweep --config <file> --batch-min <n> --batch-max <n> [--format table|csv]\n"
      << "  kvbench graph --config <file> --format mermaid\n"
      << "  kvbench version\n";
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
  try {
    Args args(argc, argv);
    if (args.values.empty() || args.values[0] == "help" || args.values[0] == "--help") {
      print_usage();
      return 0;
    }
    const std::string command = args.values[0];
    if (command == "version") {
      std::cout << "kvbench " << kVersion << "\n";
      return 0;
    }
    if (command == "simulate") return run_simulate(args);
    if (command == "compare") return run_compare(args);
    if (command == "budget") return run_budget(args);
    if (command == "sweep") return run_sweep(args);
    if (command == "graph") return run_graph(args);
    usage_error("unknown command: " + command);
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << "\n";
    return 2;
  }
}
