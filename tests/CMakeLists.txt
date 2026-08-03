include(FetchContent)
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.5.4
)
FetchContent_MakeAvailable(Catch2)

add_executable(rl_tests
  contract/environment_contract.cpp
  contract/vector_environment_contract.cpp
  test_space.cpp
  test_grid_world.cpp
  test_sync_vector_environment.cpp
  test_transition.cpp
  test_replay_buffer.cpp
  test_tabular_q_learning_agent.cpp
  test_trainer_grid_world.cpp
)

target_link_libraries(rl_tests PRIVATE rl_core Catch2::Catch2WithMain)
target_include_directories(rl_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(Catch)
catch_discover_tests(rl_tests)
