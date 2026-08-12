#include "rl/nn/actor_critic.hpp"

#include <utility>

namespace rl::nn {

ActorCriticNetwork::ActorCriticNetwork(int64_t input_dim,
                                       std::vector<int64_t> hidden_dims,
                                       int64_t num_actions,
                                       std::optional<uint64_t> seed)
    : actor_(input_dim, hidden_dims, num_actions, seed),
      critic_(input_dim, std::move(hidden_dims), 1,
              seed ? std::optional<uint64_t>(*seed + 1) : std::nullopt) {}

ActorCriticOutput ActorCriticNetwork::forward(
    const rl::tensor::Tensor& observations) {
    return ActorCriticOutput{actor_.forward(observations),
                             critic_.forward(observations)};
}

std::vector<std::shared_ptr<rl::tensor::Tensor>>
ActorCriticNetwork::parameters() const {
    auto actor_parameters = actor_.parameters();
    auto critic_parameters = critic_.parameters();
    actor_parameters.insert(actor_parameters.end(), critic_parameters.begin(),
                            critic_parameters.end());
    return actor_parameters;
}

} // namespace rl::nn
