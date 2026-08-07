cmake_minimum_required(VERSION 3.20)
project(rl_lib LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Warnings-as-errors from day one. Retrofitting a clean build onto code that
# grew up with warnings silenced is far more painful than never allowing
# them in the first place.
add_compile_options(-Wall -Wextra -Wpedantic -Werror)

add_library(rl_core
  src/core/types.cpp
  src/core/space.cpp
  src/core/environment.cpp
  src/core/vector_environment.cpp
  src/core/transition.cpp
  src/core/replay_buffer.cpp
  src/core/trainer.cpp
  src/envs/grid_world.cpp
  src/vector_envs/sync_vector_environment.cpp
  src/replay_buffers/vector_transition_storage.cpp
  src/agents/tabular_q_learning_agent.cpp
  # Milestone 5: Tensor + autograd
  src/tensor/tensor.cpp
  src/tensor/autograd.cpp
)

target_include_directories(rl_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_compile_features(rl_core PUBLIC cxx_std_20)

option(RL_BUILD_TESTS "Build the rl_lib test suite" ON)
if(RL_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
