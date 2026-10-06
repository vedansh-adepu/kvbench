#include "kvbench/memory.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

namespace {
kvbench::CacheGeometry llama() {
  kvbench::CacheGeometry g;
  g.model.num_layers = 32;
  g.model.num_attention_heads = 32;
  g.model.num_kv_heads = 8;
  g.model.head_dim = 128;
  g.model.hidden_size = 4096;
  g.quantization.dtype = kvbench::CacheDType::bf16;
  return g;
}
}

TEST_CASE("Checked arithmetic never wraps at uint64 boundaries", "[memory]") {
  using namespace kvbench;
  constexpr Count max = std::numeric_limits<Count>::max();
  REQUIRE(checked_add(max, 0) == max);
  REQUIRE(checked_mul(max, 1) == max);
  REQUIRE(checked_mul(max, 0) == 0);
  REQUIRE(checked_sub(max, max) == 0);
  REQUIRE(ceil_div(max, 2) == (Count{1} << 63));
  REQUIRE(ceil_div(max, max) == 1);
  REQUIRE(ceil_div(0, 1) == 0);
  REQUIRE_THROWS_AS(checked_add(max, 1), std::overflow_error);
  REQUIRE_THROWS_AS(checked_mul(max, 2), std::overflow_error);
  REQUIRE_THROWS_AS(checked_sub(0, 1), std::underflow_error);
  REQUIRE_THROWS_AS(ceil_div(1, 0), std::invalid_argument);
}

TEST_CASE("GiB conversion uses an exclusive representable upper bound", "[memory]") {
  using namespace kvbench;
  REQUIRE(gib_to_bytes(4) == Bytes{4} * gibibyte);
  REQUIRE(gibibyte == 1'073'741'824);
  REQUIRE(gigabyte == 1'000'000'000);
  REQUIRE(bytes_to_gib(gibibyte) == 1.0L);
  REQUIRE(gib_to_bytes(0.5L) == gibibyte / 2);
  REQUIRE(gib_to_bytes(0.5L / static_cast<long double>(gibibyte)) == 0);
  for (const auto value : {-1.0L, std::numeric_limits<long double>::infinity(), std::numeric_limits<long double>::quiet_NaN(), std::ldexp(1.0L, 34)}) {
    REQUIRE_THROWS_AS(gib_to_bytes(value), std::invalid_argument);
  }
}

TEST_CASE("Golden GQA MHA Qwen and MLA byte counts", "[golden][memory]") {
  using namespace kvbench;
  auto g = llama();
  // K and V * 32 layers * 8 KV heads * 128 dimensions * 2 bytes.
  REQUIRE(cache_layout(g).data_bytes_per_token == 131'072);
  REQUIRE(cache_layout(g).bytes_per_block == 2'097'152);
  REQUIRE(pool_block_capacity(g, gib_to_bytes(4)) == 2'048);
  REQUIRE(checked_mul(pool_block_capacity(g, gib_to_bytes(4)), g.block_size) == 32'768);
  REQUIRE(layer_blocks(g, 0, 2049) == 129);
  REQUIRE(sequence_cache_bytes(g, 2049) == Bytes{129} * 2'097'152);
  g.model.num_kv_heads = 32;
  REQUIRE(cache_layout(g).data_bytes_per_token == 524'288);
  g.model.num_layers = 80;
  g.model.num_attention_heads = 64;
  g.model.num_kv_heads = 8;
  REQUIRE(cache_layout(g).data_bytes_per_token == 327'680);
  g.model.num_layers = 61;
  g.model.attention_type = LayerKind::mla;
  g.model.kv_lora_rank = 512;
  g.model.qk_rope_head_dim = 64;
  // One latent, NOT K+V: 61 * (512+64) * bf16's 2 bytes.
  REQUIRE(cache_layout(g).data_bytes_per_token == 70'272);
  REQUIRE(cache_layout(g).bytes_per_block == Bytes{16} * 70'272);
}

TEST_CASE("MQA uses KV heads rather than attention heads", "[memory]") {
  auto g = llama();
  g.model.num_kv_heads = 1;
  REQUIRE(kvbench::cache_layout(g).data_bytes_per_token == 16'384);
  g.model.num_attention_heads = 64;
  REQUIRE(kvbench::cache_layout(g).data_bytes_per_token == 16'384);
}

TEST_CASE("Quantized data and each scale mode have exact overheads", "[memory]") {
  using namespace kvbench;
  auto g = llama();
  const auto dtype = GENERATE(CacheDType::fp8_e4m3, CacheDType::fp8_e5m2, CacheDType::int8, CacheDType::int4);
  g.quantization.dtype = dtype;
  const Bytes token_bytes = dtype == CacheDType::int4 ? 32'768 : 65'536;
  REQUIRE(cache_layout(g).data_bytes_per_token == token_bytes);
  REQUIRE(cache_layout(g).bytes_per_block == token_bytes * 16);
  g.quantization.scale_mode = ScaleMode::per_tensor;
  REQUIRE(cache_layout(g).constant_metadata_bytes == 256); // 32 layers * K/V * 4.
  REQUIRE(cache_layout(g).metadata_bytes_per_block == 0);
  REQUIRE(pool_block_capacity(g, token_bytes * 16 + 255) == 0);
  REQUIRE(pool_block_capacity(g, token_bytes * 16 + 256) == 1);
  g.quantization.scale_mode = ScaleMode::per_token_head;
  REQUIRE(cache_layout(g).metadata_bytes_per_block == 32'768); // 16*32*8*2*4.
  REQUIRE(cache_layout(g).constant_metadata_bytes == 0);
  g.quantization.zero_point_bytes = 2;
  REQUIRE(cache_layout(g).metadata_bytes_per_block == 49'152);
  g.quantization.scale_mode = ScaleMode::group;
  g.quantization.zero_point_bytes = 0;
  g.quantization.group_size = 64;
  REQUIRE(cache_layout(g).metadata_bytes_per_block == 65'536); // 32*2*(16*1024/64)*4.
  REQUIRE(cache_layout(g).bytes_per_block == token_bytes * 16 + 65'536);
}

TEST_CASE("int4 pads each cache plane and grouped scales round up", "[memory]") {
  using namespace kvbench;
  auto g = llama();
  g.model.num_layers = 1;
  g.model.num_kv_heads = 1;
  g.model.head_dim = 3;
  g.block_size = 1;
  g.quantization.dtype = CacheDType::int4;
  REQUIRE(cache_layout(g).data_bytes_per_token == 4); // ceil(3/2) for each K/V plane.
  g.block_size = 2;
  REQUIRE(cache_layout(g).data_bytes_per_block == 6); // two planes of 6 nibbles.
  g.quantization.scale_mode = ScaleMode::group;
  g.quantization.group_size = 4;
  REQUIRE(cache_layout(g).metadata_bytes_per_block == 16); // 2 * ceil(6/4) * 4.
  REQUIRE(sequence_cache_bytes(g, 3) == 44); // two 22-byte blocks.
}

TEST_CASE("MLA scales apply to one latent not two KV planes", "[memory]") {
  using namespace kvbench;
  auto g = llama();
  g.model.num_layers = 1;
  g.model.attention_type = LayerKind::mla;
  g.model.kv_lora_rank = 512;
  g.model.qk_rope_head_dim = 64;
  g.quantization.dtype = CacheDType::int8;
  g.quantization.scale_mode = ScaleMode::per_tensor;
  REQUIRE(cache_layout(g).constant_metadata_bytes == 4);
  g.quantization.scale_mode = ScaleMode::per_token_head;
  REQUIRE(cache_layout(g).metadata_bytes_per_block == 64); // one latent * 16 * 4.
}

TEST_CASE("Sliding caps and heterogeneous layer patterns are per layer", "[memory]") {
  using namespace kvbench;
  auto g = llama();
  g.model.num_layers = 2;
  g.model.num_kv_heads = 1;
  g.model.head_dim = 1;
  g.block_size = 4;
  g.model.layers = {{LayerKind::full, 0}, {LayerKind::sliding, 8}};
  REQUIRE(sequence_cache_bytes(g, 0) == 0);
  REQUIRE(layer_blocks(g, 0, 100) == 25);
  REQUIRE(layer_blocks(g, 1, 100) == 3); // ceil(8/4)+one alignment block.
  REQUIRE(sequence_cache_bytes(g, 100) == 448); // (25+3)*4 tokens*4 bytes.
  g.swa_extra_blocks = 0;
  REQUIRE(sequence_cache_bytes(g, 100) == 432);
  REQUIRE_THROWS_AS(layer_blocks(g, 2, 1), std::out_of_range);
}

TEST_CASE("Generated page counts and memory are monotonic", "[property][memory]") {
  using namespace kvbench;
  auto g = llama();
  g.block_size = GENERATE(Count{1}, Count{3}, Count{16}, Count{128});
  const Count tokens = GENERATE(Count{0}, Count{1}, Count{15}, Count{16}, Count{17}, Count{2049}, Count{65536});
  const Count expected = tokens / g.block_size + static_cast<Count>(tokens % g.block_size != 0);
  REQUIRE(layer_blocks(g, 0, tokens) == expected);
  REQUIRE(sequence_cache_bytes(g, tokens) == checked_mul(expected, cache_layout(g).bytes_per_block));
  REQUIRE(sequence_cache_bytes(g, tokens + 1) >= sequence_cache_bytes(g, tokens));
  REQUIRE(checked_mul(sequence_cache_bytes(g, tokens), 4) >= checked_mul(sequence_cache_bytes(g, tokens), 3));
}

TEST_CASE("Every memory entry validates caller-mutated geometry", "[memory][validation]") {
  using namespace kvbench;
  const int mutation = GENERATE(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11);
  auto g = llama();
  switch (mutation) {
    case 0: g.model.num_layers = 0; break;
    case 1: g.model.num_kv_heads = 0; break;
    case 2: g.model.head_dim = 0; break;
    case 3: g.block_size = 0; break;
    case 4: g.model.layers = {{LayerKind::full, 0}}; break;
    case 5: g.model.attention_type = LayerKind::sliding; break;
    case 6: g.model.attention_type = LayerKind::mla; break;
    case 7: g.quantization.dtype = static_cast<CacheDType>(999); break;
    case 8: g.quantization.scale_mode = static_cast<ScaleMode>(999); break;
    case 9: g.quantization.dtype = CacheDType::int8; g.quantization.scale_mode = ScaleMode::group; g.quantization.group_size = 0; break;
    case 10: g.quantization.scale_mode = ScaleMode::per_tensor; break;
    case 11: g.model.attention_type = static_cast<LayerKind>(999); break;
    default: FAIL("unknown fixture");
  }
  REQUIRE_THROWS_AS(cache_layout(g), std::invalid_argument);
  REQUIRE_THROWS_AS(sequence_cache_bytes(g, 1), std::invalid_argument);
  REQUIRE_THROWS_AS(layer_blocks(g, 0, 1), std::invalid_argument);
  REQUIRE_THROWS_AS(pool_block_capacity(g, gib_to_bytes(4)), std::invalid_argument);
}

TEST_CASE("Non-divisible heads warn and huge requests reject overflow", "[memory]") {
  auto g = llama();
  g.model.num_kv_heads = 7;
  REQUIRE(kvbench::validate_geometry(g).size() == 1);
  g.model.num_kv_heads = 8;
  REQUIRE(kvbench::validate_geometry(g).empty());
  REQUIRE_THROWS_AS(kvbench::sequence_cache_bytes(g, std::numeric_limits<kvbench::Count>::max()), std::overflow_error);
  REQUIRE_THROWS_AS(kvbench::parse_cache_dtype("FP16"), std::invalid_argument);
  for (const auto dtype : {kvbench::CacheDType::fp32, kvbench::CacheDType::fp16, kvbench::CacheDType::bf16, kvbench::CacheDType::fp8_e4m3, kvbench::CacheDType::fp8_e5m2, kvbench::CacheDType::int8, kvbench::CacheDType::int4}) {
    REQUIRE(kvbench::parse_cache_dtype(kvbench::cache_dtype_name(dtype)) == dtype);
  }
}
