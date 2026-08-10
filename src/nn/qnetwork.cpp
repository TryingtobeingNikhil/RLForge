#include "rl/nn/qnetwork.hpp"

#include <stdexcept>
#include <string>

namespace rl::nn {

QNetwork::QNetwork(int64_t input_dim, std::vector<int64_t> hidden_dims, int64_t output_dim) {
    // Build the full dimension sequence: [input_dim, h0, h1, ..., output_dim]
    std::vector<int64_t> all_dims;
    all_dims.reserve(hidden_dims.size() + 2);
    all_dims.push_back(input_dim);
    for (int64_t h : hidden_dims) { all_dims.push_back(h); }
    all_dims.push_back(output_dim);

    // Create (all_dims.size() - 1) Linear layers connecting adjacent dimensions.
    layers_.reserve(all_dims.size() - 1);
    for (size_t i = 0; i + 1 < all_dims.size(); ++i) {
        layers_.emplace_back(all_dims[i], all_dims[i + 1]);
    }
}

rl::tensor::Tensor QNetwork::forward(const rl::tensor::Tensor& input) {
    if (input.ndim() != 2) {
        throw std::invalid_argument(
            "QNetwork::forward: input must be 2-D [B, input_dim], got " +
            std::to_string(input.ndim()) + "-D tensor. "
            "Single-observation callers must reshape [input_dim] to [1, input_dim].");
    }

    rl::tensor::Tensor x = input;
    for (size_t i = 0; i < layers_.size(); ++i) {
        x = layers_[i].forward(x);
        // ReLU after every layer except the LAST (no activation on Q-value output).
        if (i + 1 < layers_.size()) {
            x = x.relu();
        }
    }
    return x;
}

std::vector<std::shared_ptr<rl::tensor::Tensor>> QNetwork::parameters() const {
    std::vector<std::shared_ptr<rl::tensor::Tensor>> params;
    params.reserve(2 * layers_.size());  // 2 params per Linear layer
    for (const auto& layer : layers_) {
        auto lp = layer.parameters();
        params.insert(params.end(), lp.begin(), lp.end());
    }
    return params;
}

} // namespace rl::nn
