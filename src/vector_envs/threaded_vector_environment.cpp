#include "rl/vector_envs/threaded_vector_environment.hpp"

#include <stdexcept>
#include <string>
#include <utility>

#include "rl/core/space.hpp"

namespace rl::vector_envs {

ThreadedVectorEnvironment::ThreadedVectorEnvironment(
    std::vector<EnvFactory> factories) {
    if (factories.empty()) {
        throw std::invalid_argument(
            "ThreadedVectorEnvironment requires at least one env factory");
    }

    for (const auto& factory : factories) {
        if (!factory) {
            throw std::invalid_argument("Environment factory must not be empty");
        }
    }

    try {
        workers_.reserve(factories.size());
        for (auto& factory : factories) {
            auto worker = std::make_unique<Worker>();
            worker->factory = std::move(factory);
            Worker* raw_worker = worker.get();
            worker->thread =
                std::jthread([this, raw_worker] { worker_loop(*raw_worker); });
            workers_.push_back(std::move(worker));
        }

        std::exception_ptr first_error;
        for (auto& worker : workers_) {
            try {
                wait(*worker);
            } catch (...) {
                if (!first_error) first_error = std::current_exception();
            }
        }
        if (first_error) std::rethrow_exception(first_error);

        const auto& reference_observation = observation_space();
        const auto& reference_action = action_space();
        for (size_t i = 1; i < workers_.size(); ++i) {
            if (!rl::core::spaces_compatible(
                    reference_observation,
                    workers_[i]->environment->observation_space())) {
                throw std::invalid_argument(
                    "ThreadedVectorEnvironment observation-space mismatch at sub-env " +
                    std::to_string(i));
            }
            if (!rl::core::spaces_compatible(
                    reference_action, workers_[i]->environment->action_space())) {
                throw std::invalid_argument(
                    "ThreadedVectorEnvironment action-space mismatch at sub-env " +
                    std::to_string(i));
            }
        }
    } catch (...) {
        stop_workers();
        throw;
    }
}

ThreadedVectorEnvironment::~ThreadedVectorEnvironment() { stop_workers(); }

const rl::core::Space& ThreadedVectorEnvironment::observation_space() const {
    return workers_.front()->environment->observation_space();
}

const rl::core::Space& ThreadedVectorEnvironment::action_space() const {
    return workers_.front()->environment->action_space();
}

void ThreadedVectorEnvironment::worker_loop(Worker& worker) {
    try {
        worker.environment = worker.factory();
        if (!worker.environment) {
            throw std::runtime_error("Environment factory returned nullptr");
        }
    } catch (...) {
        std::lock_guard lock(worker.mutex);
        worker.error = std::current_exception();
    }

    {
        std::lock_guard lock(worker.mutex);
        worker.completed = true;
    }
    worker.completion_cv.notify_one();

    while (true) {
        Command command;
        std::optional<uint64_t> seed;
        std::optional<rl::core::Action> action;
        {
            std::unique_lock lock(worker.mutex);
            worker.command_cv.wait(lock, [&worker] {
                return worker.command != Command::None;
            });
            command = worker.command;
            seed = worker.seed;
            action = worker.action;
            worker.command = Command::None;
        }

        if (command == Command::Stop) {
            worker.environment.reset();
            return;
        }

        try {
            if (!worker.environment) {
                throw std::runtime_error("Environment worker failed to initialize");
            }
            if (command == Command::Reset) {
                auto result = worker.environment->reset(seed);
                std::lock_guard lock(worker.mutex);
                worker.reset_result = std::move(result);
            } else {
                if (!action) throw std::logic_error("Step command has no action");
                auto result = worker.environment->step(*action);
                if (result.terminated || result.truncated) {
                    auto terminal_observation = result.observation;
                    auto reset = worker.environment->reset();
                    result.observation = std::move(reset.observation);
                    // The vector assembly retrieves this separately through
                    // reset_result, avoiding any change to StepResult.
                    rl::core::ResetResult terminal;
                    terminal.observation = std::move(terminal_observation);
                    std::lock_guard lock(worker.mutex);
                    worker.reset_result = std::move(terminal);
                    worker.step_result = std::move(result);
                } else {
                    std::lock_guard lock(worker.mutex);
                    worker.reset_result.reset();
                    worker.step_result = std::move(result);
                }
            }
        } catch (...) {
            std::lock_guard lock(worker.mutex);
            worker.error = std::current_exception();
        }

        {
            std::lock_guard lock(worker.mutex);
            worker.completed = true;
        }
        worker.completion_cv.notify_one();
    }
}

void ThreadedVectorEnvironment::dispatch(Worker& worker, Command command) {
    std::lock_guard lock(worker.mutex);
    worker.completed = false;
    worker.error = nullptr;
    worker.reset_result.reset();
    worker.step_result.reset();
    worker.command = command;
    worker.command_cv.notify_one();
}

void ThreadedVectorEnvironment::wait(Worker& worker) {
    std::unique_lock lock(worker.mutex);
    worker.completion_cv.wait(lock, [&worker] { return worker.completed; });
    if (worker.error) std::rethrow_exception(worker.error);
}

void ThreadedVectorEnvironment::stop_workers() noexcept {
    for (auto& worker : workers_) {
        if (!worker->thread.joinable()) continue;
        {
            std::lock_guard lock(worker->mutex);
            worker->command = Command::Stop;
        }
        worker->command_cv.notify_one();
    }
    for (auto& worker : workers_) {
        if (worker->thread.joinable()) worker->thread.join();
    }
}

rl::core::VectorResetResult ThreadedVectorEnvironment::reset_impl(
    std::optional<uint64_t> seed) {
    for (size_t i = 0; i < workers_.size(); ++i) {
        {
            std::lock_guard lock(workers_[i]->mutex);
            workers_[i]->seed = seed ? std::optional<uint64_t>(*seed + i) : std::nullopt;
            workers_[i]->action.reset();
        }
        dispatch(*workers_[i], Command::Reset);
    }
    std::exception_ptr first_error;
    for (auto& worker : workers_) {
        try {
            wait(*worker);
        } catch (...) {
            if (!first_error) first_error = std::current_exception();
        }
    }
    if (first_error) std::rethrow_exception(first_error);

    rl::core::VectorResetResult result;
    result.observations.reserve(workers_.size());
    result.infos.reserve(workers_.size());
    for (auto& worker : workers_) {
        std::lock_guard lock(worker->mutex);
        result.observations.push_back(std::move(worker->reset_result->observation));
        result.infos.push_back(std::move(worker->reset_result->info));
        worker->reset_result.reset();
    }
    return result;
}

rl::core::VectorStepResult ThreadedVectorEnvironment::step_impl(
    const std::vector<rl::core::Action>& actions) {
    if (actions.size() != workers_.size()) {
        throw std::invalid_argument(
            "ThreadedVectorEnvironment::step action count mismatch");
    }
    for (size_t i = 0; i < workers_.size(); ++i) {
        {
            std::lock_guard lock(workers_[i]->mutex);
            workers_[i]->action = actions[i];
            workers_[i]->seed.reset();
        }
        dispatch(*workers_[i], Command::Step);
    }
    std::exception_ptr first_error;
    for (auto& worker : workers_) {
        try {
            wait(*worker);
        } catch (...) {
            if (!first_error) first_error = std::current_exception();
        }
    }
    if (first_error) std::rethrow_exception(first_error);

    const size_t n = workers_.size();
    rl::core::VectorStepResult result;
    result.observations.resize(n);
    result.rewards.resize(n);
    result.terminated.resize(n);
    result.truncated.resize(n);
    result.infos.resize(n);
    result.final_observations.resize(n);
    for (size_t i = 0; i < n; ++i) {
        std::lock_guard lock(workers_[i]->mutex);
        auto& step = *workers_[i]->step_result;
        result.observations[i] = std::move(step.observation);
        result.rewards[i] = step.reward;
        result.terminated[i] = step.terminated ? 1 : 0;
        result.truncated[i] = step.truncated ? 1 : 0;
        result.infos[i] = std::move(step.info);
        if (workers_[i]->reset_result) {
            result.final_observations[i] =
                std::move(workers_[i]->reset_result->observation);
        }
        workers_[i]->step_result.reset();
        workers_[i]->reset_result.reset();
        workers_[i]->action.reset();
    }
    return result;
}

} // namespace rl::vector_envs
