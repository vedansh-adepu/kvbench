#include "kvbench/output.hpp"
#include <catch2/catch_test_macros.hpp>

TEST_CASE("Output labels cannot inject Markdown or Mermaid statements", "[output]") {
  const auto escaped = kvbench::escape_mermaid("x\"]\nclick evil | & < >\\");
  REQUIRE(escaped == "x&#34;&#93;&#10;click evil &#124; &#38; &#60; &#62;&#92;");
  REQUIRE(escaped.find('\n') == std::string::npos);
  REQUIRE_THROWS(kvbench::render_report({{"model_name","name"}},"invalid"));
  REQUIRE(kvbench::parse_json(kvbench::render_report({{"kvbench_result",2},{"count",kvbench::Count{18446744073709551615ULL}}},"json")).at("count").get<kvbench::Count>() == 18446744073709551615ULL);
}
