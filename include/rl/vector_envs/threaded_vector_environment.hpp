#pragma once

#include <condition_variable>
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

#include "rl/core/environment.hpp"
#include "rl/core/vector_environment.hpp"
#include "rl/vector_envs/sync_vector_environment.hpp"

namespace rl::vector_envs {

// One persistent worker per environment. Each environment is constructed,
// stepped, reset, and destroyed by its worker thread. Batched calls execute
// concurrently and return in deterministic factory order.
class ThreadedVectorEnvironment final : public rl::core::VectorEnvironmentBase {
public:
    explicit ThreadedVectorEnvironment(std::vector<EnvFactory> factories);
    ~ThreadedVectorEnvironment() override;

    ThreadedVectorEnvironment(const ThreadedVectorEnvironment&) = delete;
    ThreadedVectorEnvironment& operator=(const ThreadedVectorEnvironment&) = delete;

    size_t num_envs() const override { return workers_.size(); }
    const rl::core::Space& observation_space() const override;
    const rl::core::Space& action_space() const override;

protected:
    rl::core::VectorResetResult reset_impl(std::optional<uint64_t> seed) override;
    rl::core::VectorStepResult step_impl(
        const std::vector<rl::core::Action>& actions) override;

private:
    enum class Command { None, Reset, Step, Stop };

    struct Worker {
        EnvFactory factory;
        std::unique_ptr<rl::core::Environment> environment;
        std::jthread thread;
        std::mutex mutex;
        std::condition_variable command_cv;
        std::condition_variable completion_cv;
        Command command = Command::None;
        bool completed = false;
        std::optional<uint64_t> seed;
        std::optional<rl::core::Action> action;
        std::optional<rl::core::ResetResult> reset_result;
        std::optional<rl::core::StepResult> step_result;
        std::exception_ptr error;
    };

    void worker_loop(Worker& worker);
    void dispatch(Worker& worker, Command command);
    void wait(Worker& worker);
    void stop_workers() noexcept;

    std::vector<std::unique_ptr<Worker>> workers_;
};

} // namespace rl::vector_envs
