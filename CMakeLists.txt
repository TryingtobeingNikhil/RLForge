cmake_minimum_required(VERSION 3.20)
project(rl_lib LANGUAGES CXX)

find_package(Threads REQUIRED)

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

# ---------------------------------------------------------------------------
# Sanitizer support — opt-in via -DRL_SANITIZE=ON.
#
# Usage:
#   cmake -S . -B build-asan -DCMAKE_BUILD_TYPE=Debug -DRL_SANITIZE=ON
#   cmake --build build-asan -j4
#   cd build-asan && ctest --output-on-failure
#
# Enables AddressSanitizer + LeakSanitizer (LSan is bundled with ASan on
# Linux and macOS). On macOS, LSan requires SIP to be partially relaxed;
# if that is not available, set ASAN_OPTIONS=detect_leaks=0 to skip it.
# ---------------------------------------------------------------------------
option(RL_SANITIZE "Build with AddressSanitizer + LeakSanitizer" OFF)
if(RL_SANITIZE)
  message(STATUS "[RLForge] Sanitizers: AddressSanitizer + LeakSanitizer enabled")
  add_compile_options(-fsanitize=address -fno-omit-frame-pointer -g)
  add_link_options(-fsanitize=address)
endif()

# Optional Milestone 10 acceleration backends. The portable CPU backend is
# always available and remains the default.
option(RL_ENABLE_BLAS "Enable the CBLAS tensor backend" OFF)
option(RL_ENABLE_CUDA "Enable the CUDA/cuBLAS tensor backend" OFF)

if(RL_ENABLE_BLAS)
  if(APPLE)
    find_library(RL_CBLAS_LIBRARY Accelerate REQUIRED)
  else()
    find_path(RL_CBLAS_INCLUDE_DIR cblas.h REQUIRED)
    find_library(RL_CBLAS_LIBRARY NAMES openblas cblas blas REQUIRED)
  endif()
endif()

if(RL_ENABLE_CUDA)
  find_package(CUDAToolkit REQUIRED)
endif()

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
  src/vector_envs/threaded_vector_environment.cpp
  src/replay_buffers/vector_transition_storage.cpp
  src/replay_buffers/rollout_buffer.cpp
  src/agents/tabular_q_learning_agent.cpp
  # Milestone 5: Tensor + autograd
  src/tensor/tensor.cpp
  src/tensor/autograd.cpp
  src/tensor/backend.cpp
  src/tensor/blas_backend.cpp
  src/tensor/cuda_backend.cpp
  # Milestone 6: NN layers + optimizers
  src/nn/linear.cpp
  src/optim/optimizer.cpp
  src/optim/sgd.cpp
  src/optim/adam.cpp
  # Milestone 7: DQN
  src/nn/qnetwork.cpp
  src/agents/dqn.cpp
  # Milestone 8: PPO
  src/nn/actor_critic.cpp
  src/agents/ppo.cpp
)

target_include_directories(rl_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_compile_features(rl_core PUBLIC cxx_std_20)
target_link_libraries(rl_core PUBLIC Threads::Threads)

if(RL_ENABLE_BLAS)
  target_compile_definitions(rl_core PRIVATE RL_HAS_CBLAS=1)
  if(APPLE)
    target_compile_definitions(rl_core PRIVATE RL_USE_ACCELERATE=1)
  else()
    target_include_directories(rl_core PRIVATE ${RL_CBLAS_INCLUDE_DIR})
  endif()
  target_link_libraries(rl_core PRIVATE ${RL_CBLAS_LIBRARY})
endif()

if(RL_ENABLE_CUDA)
  target_compile_definitions(rl_core PRIVATE RL_HAS_CUDA=1)
  target_link_libraries(rl_core PRIVATE CUDA::cudart CUDA::cublas)
endif()

option(RL_BUILD_TESTS "Build the rl_lib test suite" ON)
if(RL_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
