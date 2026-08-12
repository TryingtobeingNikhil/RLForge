#include "rl/nn/linear.hpp"

#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>

namespace rl::nn {

Linear::Linear(int64_t in_features, int64_t out_features) {
    std::mt19937_64 rng{std::random_device{}()};
    initialize(in_features, out_features, rng);
}

Linear::Linear(int64_t in_features, int64_t out_features,
               std::mt19937_64& rng) {
    initialize(in_features, out_features, rng);
}

void Linear::initialize(int64_t in_features, int64_t out_features,
                        std::mt19937_64& rng) {
    if (in_features <= 0 || out_features <= 0) {
        throw std::invalid_argument(
            "Linear in_features and out_features must be positive");
    }
    if (out_features > std::numeric_limits<int64_t>::max() / in_features) {
        throw std::overflow_error("Linear weight shape is too large");
    }
    // -----------------------------------------------------------------
    // Weight initialisation: He (Kaiming) normal.
    //   Sample from N(0, sigma^2) where sigma = sqrt(2 / in_features).
    //   This is the standard initialisation for layers followed by ReLU,
    //   ensuring variance of activations is preserved at initialisation.
    // -----------------------------------------------------------------
    const double sigma = std::sqrt(2.0 / static_cast<double>(in_features));

    std::normal_distribution<double> dist(0.0, sigma);

    const int64_t w_numel = out_features * in_features;
    std::vector<double> w_data(static_cast<size_t>(w_numel));
    for (auto& v : w_data) { v = dist(rng); }

    weight_ = std::make_shared<rl::tensor::Tensor>(
        rl::tensor::Tensor::from_data(std::move(w_data), {out_features, in_features}));
    weight_->requires_grad_(true);

    // -----------------------------------------------------------------
    // Bias initialisation: zero.
    // -----------------------------------------------------------------
    bias_ = std::make_shared<rl::tensor::Tensor>(
        rl::tensor::Tensor::zeros({out_features}));
    bias_->requires_grad_(true);
}

rl::tensor::Tensor Linear::forward(const rl::tensor::Tensor& input) {
    // x:[B, in_features] @ W^T:[in_features, out_features] → [B, out_features]
    // Then + bias_:[out_features]  (broadcasting add, [B,out] + [out])
    return input.matmul(weight_->transpose()).add(*bias_);
}

std::vector<std::shared_ptr<rl::tensor::Tensor>> Linear::parameters() const {
    return {weight_, bias_};
}

} // namespace rl::nn
