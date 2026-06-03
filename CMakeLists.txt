cmake_minimum_required(VERSION 3.16)

project(kvbench VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(kvbench_core
  src/config.cpp
  src/estimator.cpp
  src/json.cpp
  src/scheduler.cpp
)

target_include_directories(kvbench_core PUBLIC include)

target_compile_options(kvbench_core PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>
)

add_executable(kvbench src/main.cpp)
target_link_libraries(kvbench PRIVATE kvbench_core)
target_compile_options(kvbench PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>
)

enable_testing()

add_executable(kvbench_tests tests/test_kvbench.cpp)
target_link_libraries(kvbench_tests PRIVATE kvbench_core)
target_compile_options(kvbench_tests PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>
)

add_test(NAME kvbench_tests COMMAND kvbench_tests)
