#include <catch2/catch_test_macros.hpp>

#include "kvbench/output.hpp"

TEST_CASE("Output labels cannot inject Markdown or Mermaid statements", "[output]") {
  const auto escaped = kvbench::escape_mermaid("x\"]\nclick evil | & < >\\");
  REQUIRE(escaped == "x&#34;&#93;&#10;click evil &#124; &#38; &#60; &#62;&#92;");
  REQUIRE(escaped.find('\n') == std::string::npos);
  REQUIRE_THROWS(kvbench::render_report({{"model_name", "name"}}, "invalid"));
  REQUIRE(
      kvbench::parse_json(
          kvbench::render_report(
              {{"kvbench_result", 2}, {"count", kvbench::Count{18446744073709551615ULL}}}, "json"))
          .at("count")
          .get<kvbench::Count>() == 18446744073709551615ULL);
}

TEST_CASE("Human capacity output labels explicitly bounded lower limits", "[output]") {
  kvbench::Json report = {
      {"static_worst_case", {{"max_concurrency", 10}, {"max_concurrency_is_lower_bound", true}}}};
  REQUIRE(kvbench::render_report(report, "text").find("at least 10") != std::string::npos);
  REQUIRE(kvbench::parse_json(kvbench::render_report(report, "json"))
              .at("static_worst_case")
              .at("max_concurrency") == 10);
  report["static_worst_case"]["max_concurrency_is_lower_bound"] = false;
  REQUIRE(kvbench::render_report(report, "text").find("at least") == std::string::npos);
}
