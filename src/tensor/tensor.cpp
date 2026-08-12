#include "rl/tensor/tensor.hpp"

#include "rl/tensor/backend.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <sstream>
#include <stack>
#include <stdexcept>
#include <unordered_set>

namespace rl::tensor {

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

namespace {

int64_t shape_numel(const std::vector<int64_t>& shape) {
    int64_t n = 1;
    for (int64_t d : shape) {
        if (d < 0) {
            throw std::invalid_argument(
                "Tensor shape dimensions must be non-negative");
        }
        if (d != 0 && n > std::numeric_limits<int64_t>::max() / d) {
            throw std::overflow_error("Tensor shape product overflows int64_t");
        }
        n *= d;
    }
    return n;
}

std::string shape_str(const std::vector<int64_t>& shape) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << shape[i];
    }
    oss << "]";
    return oss.str();
}

void check_same_shape(const Tensor& a, const Tensor& b, const char* op_name) {
    if (a.shape() != b.shape()) {
        throw std::invalid_argument(
            std::string(op_name) + ": shape mismatch — lhs has shape " +
            shape_str(a.shape()) + " but rhs has shape " + shape_str(b.shape()) +
            ". Tensor-tensor broadcasting is not supported for this op.");
    }
}

// ---------------------------------------------------------------------------
// VERSION CHECK helper — called inside backward closures that read captured
// input-tensor storage values. Throws std::runtime_error on mismatch.
//
// Parameters:
//   storage      — the shared_ptr<Storage> captured at forward time.
//   saved_version— the storage->version captured at forward (closure-creation) time.
//   op_name      — human-readable op name for the error message.
//   which        — "lhs"/"rhs"/"input" to identify which operand is stale.
// ---------------------------------------------------------------------------
void check_version(const std::shared_ptr<Storage>& storage,
                   int64_t saved_version,
                   const char* op_name,
                   const char* which) {
    if (storage->version != saved_version) {
        throw std::runtime_error(
            std::string(op_name) + " backward: " + which +
            " tensor was mutated in-place after the forward pass that created "
            "this graph node — stale gradient computation detected. "
            "(saved version=" + std::to_string(saved_version) +
            ", current version=" + std::to_string(storage->version) + ")");
    }
}

// ---------------------------------------------------------------------------
// Topological sort: iterative post-order DFS, root-first output.
// ---------------------------------------------------------------------------
std::vector<std::shared_ptr<Node>> topological_order(const std::shared_ptr<Node>& root) {
    std::vector<std::shared_ptr<Node>> order;
    std::unordered_set<Node*> visited;

    std::stack<std::pair<std::shared_ptr<Node>, size_t>> stk;
    stk.push({root, 0});

    while (!stk.empty()) {
        auto& [node, idx] = stk.top();
        if (idx < node->parents.size()) {
            auto& parent = node->parents[idx];
            ++idx;
            if (parent && visited.find(parent.get()) == visited.end()) {
                stk.push({parent, 0});
            }
        } else {
            if (visited.find(node.get()) == visited.end()) {
                visited.insert(node.get());
                order.push_back(node);
            }
            stk.pop();
        }
    }

    std::reverse(order.begin(), order.end());
    return order;
}

// ---------------------------------------------------------------------------
// Distribute gradient from a backward_fn to a parent node's incoming_grad.
//
// Uses direct storage access (NOT data_mutable()) to avoid incrementing the
// version counter on intermediate gradient accumulation tensors — those tensors
// are freshly allocated scratch buffers, not user-facing tensors with graphs.
// ---------------------------------------------------------------------------
void distribute_grad(std::shared_ptr<Node>& parent_node, const Tensor& grad_contribution) {
    if (!parent_node) return;
    if (!parent_node->incoming_grad) {
        parent_node->incoming_grad =
            std::make_shared<Tensor>(grad_contribution.shape());
    }
    // data_mutable() bumps the version of the incoming_grad scratch tensor.
    // This is harmless: no backward closure ever captures the version of a
    // freshly-allocated intermediate gradient accumulation buffer.
    auto dst = parent_node->incoming_grad->data_mutable();
    const auto& src = grad_contribution.data();
    for (size_t i = 0; i < dst.size(); ++i) {
        dst[i] += src[i];
    }
}

// ---------------------------------------------------------------------------
// Core backward implementation.
// ---------------------------------------------------------------------------
void run_backward(const std::shared_ptr<Node>& root_node, Tensor initial_grad) {
    if (!root_node) return;

    auto order = topological_order(root_node);

    // Clear incoming_grad on ALL reachable nodes before starting this pass.
    for (auto& node : order) {
        node->incoming_grad = nullptr;
    }

    root_node->incoming_grad = std::make_shared<Tensor>(std::move(initial_grad));

    for (auto& node : order) {
        if (!node->incoming_grad || !node->backward_fn) {
            continue;
        }
        // Run every backward_fn under no_grad().
        auto guard = no_grad();
        node->backward_fn(*node->incoming_grad);
    }
}


} // namespace

// ---------------------------------------------------------------------------
// Tensor::make_output
// ---------------------------------------------------------------------------
Tensor Tensor::make_output(std::shared_ptr<Storage> storage, std::vector<int64_t> shape,
                            bool rg, std::shared_ptr<Node> node) {
    Tensor t(shape);
    t.storage_ = std::move(storage);
    t.rg_ = rg;
    t.node_ = std::move(node);
    return t;
}

// ---------------------------------------------------------------------------
// make_leaf_node: create a terminal Node for a leaf tensor that requires grad.
//
// Key design: leaf tensors with requires_grad=true get a Node whose
// backward_fn accumulates into their grad buffer. Storing a shared_ptr to
// the grad buffer avoids the dangling-pointer problem that would occur if we
// stored a raw Tensor* pointer (which would dangle if the leaf Tensor is
// passed by value).
// ---------------------------------------------------------------------------
std::shared_ptr<Node> Tensor::make_leaf_node() {
    // Ensure grad_ buffer exists.
    if (!grad_) {
        grad_ = std::make_shared<Tensor>(shape_);
    }
    // Capture grad_ by value (shared_ptr copy) — safe even if the Tensor
    // object itself moves or goes out of scope.
    auto grad_ptr = grad_;
    auto leaf_node = std::make_shared<Node>();
    leaf_node->backward_fn = [grad_ptr](const Tensor& incoming) {
        // Accumulate into the leaf's grad buffer.
        // data_mutable() bumps the grad buffer's version, which is harmless:
        // no closure captures the grad buffer's version (it is a separate
        // Tensor from the leaf tensor itself — see docs/tensor_autograd_api.md).
        auto dst = grad_ptr->data_mutable();
        const auto& src = incoming.data();
        for (size_t i = 0; i < dst.size(); ++i) {
            dst[i] += src[i];
        }
    };
    // Leaf nodes have no parents.
    return leaf_node;
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

Tensor::Tensor(std::vector<int64_t> shape)
    : storage_(std::make_shared<Storage>(shape_numel(shape))), shape_(std::move(shape)) {}

Tensor::Tensor(std::vector<double> data_in, std::vector<int64_t> shape) {
    int64_t expected = shape_numel(shape);
    if (static_cast<int64_t>(data_in.size()) != expected) {
        throw std::invalid_argument(
            "Tensor constructor: data.size() (" + std::to_string(data_in.size()) +
            ") does not match product of shape " + shape_str(shape) + " (" +
            std::to_string(expected) + ").");
    }
    storage_ = std::make_shared<Storage>(std::move(data_in));
    shape_ = std::move(shape);
}

// ---------------------------------------------------------------------------
// Static factories
// ---------------------------------------------------------------------------

Tensor Tensor::zeros(std::vector<int64_t> shape) { return Tensor(shape); }

Tensor Tensor::ones(std::vector<int64_t> shape) {
    Tensor t(shape);
    std::fill(t.storage_->data.begin(), t.storage_->data.end(), 1.0);
    return t;
}

Tensor Tensor::from_data(std::vector<double> data_in, std::vector<int64_t> shape) {
    return Tensor(std::move(data_in), std::move(shape));
}

double Tensor::operator[](int64_t i) const {
    if (i < 0 || i >= numel()) {
        throw std::out_of_range("Tensor index out of range");
    }
    return storage_->data[static_cast<size_t>(i)];
}

// ---------------------------------------------------------------------------
// Scalar access
// ---------------------------------------------------------------------------

double Tensor::item() const {
    if (numel() != 1) {
        throw std::invalid_argument(
            "Tensor::item() called on a tensor with " + std::to_string(numel()) +
            " elements (shape " + shape_str(shape_) + "). Only numel()==1 tensors "
            "support item().");
    }
    return storage_->data[0];
}

// ---------------------------------------------------------------------------
// requires_grad
// ---------------------------------------------------------------------------

Tensor& Tensor::requires_grad_(bool rg) {
    rg_ = rg;
    if (rg_ && !node_) {
        // Create/update the leaf node so future op uses can trace back here.
        node_ = make_leaf_node();
    }
    return *this;
}

// ---------------------------------------------------------------------------
// Gradient accumulation (public; also used by leaf_node closure)
// ---------------------------------------------------------------------------

void Tensor::accumulate_grad(const Tensor& g) {
    if (!rg_) return;
    if (g.shape() != shape_) {
        throw std::invalid_argument(
            "Tensor::accumulate_grad requires matching gradient shape");
    }
    if (!grad_) {
        grad_ = std::make_shared<Tensor>(shape_);
    }
    auto& dst = grad_->storage_->data;
    const auto& src = g.storage_->data;
    for (size_t i = 0; i < dst.size(); ++i) {
        dst[i] += src[i];
    }
}

void Tensor::zero_grad() {
    if (grad_) {
        std::fill(grad_->storage_->data.begin(), grad_->storage_->data.end(), 0.0);
    }
}

// ---------------------------------------------------------------------------
// backward
// ---------------------------------------------------------------------------

void Tensor::backward() {
    if (numel() != 1) {
        throw std::invalid_argument(
            "Tensor::backward() called on a non-scalar tensor (shape " +
            shape_str(shape_) + ", numel=" + std::to_string(numel()) + "). "
            "Use backward(Tensor grad) to supply an explicit upstream gradient.");
    }
    Tensor seed({});
    seed.storage_->data[0] = 1.0;
    run_backward(node_, std::move(seed));
}

void Tensor::backward(Tensor grad) {
    if (grad.shape() != shape_) {
        throw std::invalid_argument(
            "Tensor::backward(grad): grad shape " + shape_str(grad.shape()) +
            " does not match tensor shape " + shape_str(shape_) + ".");
    }
    run_backward(node_, std::move(grad));
}

// ---------------------------------------------------------------------------
// detach
// ---------------------------------------------------------------------------

Tensor Tensor::detach() const {
    Tensor t(shape_);
    t.storage_ = storage_;
    t.rg_   = false;
    t.node_ = nullptr;
    return t;
}

// ---------------------------------------------------------------------------
// Helper: get the "input node" for an operand in a binary/unary op.
// ---------------------------------------------------------------------------
namespace {

inline std::shared_ptr<Node> input_node(const Tensor& t) {
    return t.requires_grad() ? t.node() : nullptr;
}

} // namespace

// ---------------------------------------------------------------------------
// add
//
// Broadcasting rule (Milestone 6):
//   (a) Exact shape match: elementwise addition, gradient = upstream gradient.
//   (b) [B,N] + [N]: row-wise broadcast of the [N] operand across batch B.
//       Backward for the [N] operand: sum upstream gradient over the B rows.
//   Any other combination throws std::invalid_argument.
// ---------------------------------------------------------------------------

Tensor Tensor::add(const Tensor& other) const {
    const bool exact_match = (shape_ == other.shape_);

    // Check for [B,N]+[N] broadcast case.
    const bool broadcast_bias =
        !exact_match &&
        ndim() == 2 &&
        other.ndim() == 1 &&
        shape_[1] == other.shape_[0];

    if (!exact_match && !broadcast_bias) {
        throw std::invalid_argument(
            "Tensor::add: unsupported shape combination — lhs has shape " +
            shape_str(shape_) + " but rhs has shape " + shape_str(other.shape_) +
            ". Supported: (a) exact shape match, or (b) [B,N]+[N] bias broadcast.");
    }

    const int64_t total = numel();
    auto out_storage = std::make_shared<Storage>(total);

    if (exact_match) {
        for (int64_t i = 0; i < total; ++i) {
            out_storage->data[static_cast<size_t>(i)] =
                storage_->data[static_cast<size_t>(i)] +
                other.storage_->data[static_cast<size_t>(i)];
        }
    } else {
        // [B,N] + [N]: bias[j] is added to every row i.
        const int64_t B = shape_[0], N = shape_[1];
        for (int64_t i = 0; i < B; ++i) {
            for (int64_t j = 0; j < N; ++j) {
                out_storage->data[static_cast<size_t>(i * N + j)] =
                    storage_->data[static_cast<size_t>(i * N + j)] +
                    other.storage_->data[static_cast<size_t>(j)];
            }
        }
    }

    bool out_rg = grad_mode_enabled() && (rg_ || other.rg_);
    std::shared_ptr<Node> out_node;

    if (out_rg) {
        auto self_nd  = input_node(*this);
        auto other_nd = input_node(other);

        out_node = std::make_shared<Node>();
        if (self_nd)  out_node->parents.push_back(self_nd);
        if (other_nd) out_node->parents.push_back(other_nd);

        if (exact_match) {
            // Gradient passes through unchanged for both operands.
            out_node->backward_fn = [self_nd, other_nd](const Tensor& grad) mutable {
                if (self_nd)  distribute_grad(self_nd, grad);
                if (other_nd) distribute_grad(other_nd, grad);
            };
        } else {
            // [B,N] + [N] broadcast backward.
            // d(loss)/d(self)[i,j] = upstream[i,j]  — same shape, pass through.
            // d(loss)/d(bias)[j]   = sum_i upstream[i,j]  — sum over batch.
            const int64_t B = shape_[0], N = shape_[1];
            const std::vector<int64_t> bias_shape = other.shape_;

            out_node->backward_fn = [self_nd, other_nd, B, N,
                                     bias_shape](const Tensor& grad) mutable {
                if (self_nd) {
                    // Gradient for [B,N] operand: pass through unchanged.
                    distribute_grad(self_nd, grad);
                }
                if (other_nd) {
                    // Gradient for [N] operand: sum over the batch dimension.
                    Tensor g_bias(bias_shape);
                    for (int64_t i = 0; i < B; ++i) {
                        for (int64_t j = 0; j < N; ++j) {
                            g_bias[j] += grad[i * N + j];
                        }
                    }
                    distribute_grad(other_nd, g_bias);
                }
            };
        }
    }

    return make_output(std::move(out_storage), shape_, out_rg, std::move(out_node));
}

// ---------------------------------------------------------------------------
// sub
// ---------------------------------------------------------------------------

Tensor Tensor::sub(const Tensor& other) const {
    check_same_shape(*this, other, "Tensor::sub");

    auto out_storage = std::make_shared<Storage>(numel());
    for (int64_t i = 0; i < numel(); ++i) {
        out_storage->data[static_cast<size_t>(i)] =
            storage_->data[static_cast<size_t>(i)] -
            other.storage_->data[static_cast<size_t>(i)];
    }

    bool out_rg = grad_mode_enabled() && (rg_ || other.rg_);
    std::shared_ptr<Node> out_node;

    if (out_rg) {
        auto self_nd  = input_node(*this);
        auto other_nd = input_node(other);
        auto sh       = shape_;

        out_node = std::make_shared<Node>();
        if (self_nd)  out_node->parents.push_back(self_nd);
        if (other_nd) out_node->parents.push_back(other_nd);

        out_node->backward_fn = [self_nd, other_nd, sh](const Tensor& grad) mutable {
            if (self_nd) distribute_grad(self_nd, grad);
            if (other_nd) {
                Tensor neg_grad(sh);
                for (int64_t i = 0; i < grad.numel(); ++i) {
                    neg_grad[i] = -grad[i];
                }
                distribute_grad(other_nd, neg_grad);
            }
        };
    }

    return make_output(std::move(out_storage), shape_, out_rg, std::move(out_node));
}

// ---------------------------------------------------------------------------
// mul (elementwise)
//
// VERSION GUARD: this backward closure reads other_storage and self_storage
// values at backward time — must check versions.
// ---------------------------------------------------------------------------

Tensor Tensor::mul(const Tensor& other) const {
    check_same_shape(*this, other, "Tensor::mul");

    auto out_storage = std::make_shared<Storage>(numel());
    for (int64_t i = 0; i < numel(); ++i) {
        out_storage->data[static_cast<size_t>(i)] =
            storage_->data[static_cast<size_t>(i)] *
            other.storage_->data[static_cast<size_t>(i)];
    }

    bool out_rg = grad_mode_enabled() && (rg_ || other.rg_);
    std::shared_ptr<Node> out_node;

    if (out_rg) {
        auto self_storage  = storage_;       // data kept alive for backward
        auto other_storage = other.storage_;
        auto self_nd       = input_node(*this);
        auto other_nd      = input_node(other);
        auto sh            = shape_;
        int64_t n          = numel();
        // Capture versions at forward time for the version guard.
        int64_t self_ver   = storage_->version;
        int64_t other_ver  = other.storage_->version;

        out_node = std::make_shared<Node>();
        if (self_nd)  out_node->parents.push_back(self_nd);
        if (other_nd) out_node->parents.push_back(other_nd);

        out_node->backward_fn = [self_nd, other_nd, self_storage, other_storage,
                                  sh, n, self_ver, other_ver](const Tensor& grad) mutable {
            if (self_nd) {
                check_version(other_storage, other_ver, "mul", "rhs");
                Tensor g(sh);
                for (int64_t i = 0; i < n; ++i) {
                    g[i] = grad[i] * other_storage->data[static_cast<size_t>(i)];
                }
                distribute_grad(self_nd, g);
            }
            if (other_nd) {
                check_version(self_storage, self_ver, "mul", "lhs");
                Tensor g(sh);
                for (int64_t i = 0; i < n; ++i) {
                    g[i] = grad[i] * self_storage->data[static_cast<size_t>(i)];
                }
                distribute_grad(other_nd, g);
            }
        };
    }

    return make_output(std::move(out_storage), shape_, out_rg, std::move(out_node));
}

// ---------------------------------------------------------------------------
// mul (scalar)
// ---------------------------------------------------------------------------

Tensor Tensor::mul(double scalar) const {
    auto out_storage = std::make_shared<Storage>(numel());
    for (int64_t i = 0; i < numel(); ++i) {
        out_storage->data[static_cast<size_t>(i)] =
            storage_->data[static_cast<size_t>(i)] * scalar;
    }

    bool out_rg = grad_mode_enabled() && rg_;
    std::shared_ptr<Node> out_node;

    if (out_rg) {
        auto self_nd = input_node(*this);
        auto sh      = shape_;
        int64_t n    = numel();

        out_node = std::make_shared<Node>();
        if (self_nd) out_node->parents.push_back(self_nd);

        out_node->backward_fn = [self_nd, sh, scalar, n](const Tensor& grad) mutable {
            if (self_nd) {
                Tensor g(sh);
                for (int64_t i = 0; i < n; ++i) {
                    g[i] = grad[i] * scalar;
                }
                distribute_grad(self_nd, g);
            }
        };
    }

    return make_output(std::move(out_storage), shape_, out_rg, std::move(out_node));
}

// ---------------------------------------------------------------------------
// matmul (2-D only)
//
// VERSION GUARD: backward closure reads self_storage and other_storage values.
// ---------------------------------------------------------------------------

Tensor Tensor::matmul(const Tensor& other) const {
    if (ndim() != 2) {
        throw std::invalid_argument(
            "Tensor::matmul: lhs must be 2-D (got shape " + shape_str(shape_) + ").");
    }
    if (other.ndim() != 2) {
        throw std::invalid_argument(
            "Tensor::matmul: rhs must be 2-D (got shape " + shape_str(other.shape_) + ").");
    }

    const int64_t M = shape_[0], K = shape_[1];
    const int64_t K2 = other.shape_[0], N = other.shape_[1];

    if (M <= 0 || K <= 0 || K2 <= 0 || N <= 0) {
        throw std::invalid_argument(
            "Tensor::matmul requires non-empty matrix dimensions");
    }

    if (K != K2) {
        throw std::invalid_argument(
            "Tensor::matmul: inner dimension mismatch: lhs shape " + shape_str(shape_) +
            " has K=" + std::to_string(K) + " but rhs shape " +
            shape_str(other.shape_) + " has rows=" + std::to_string(K2) + ".");
    }

    auto out_storage = std::make_shared<Storage>(shape_numel({M, N}));
    auto backend = current_backend();
    backend->matmul(storage_->data.data(), other.storage_->data.data(),
                    out_storage->data.data(), M, K, N);

    bool out_rg = grad_mode_enabled() && (rg_ || other.rg_);
    std::shared_ptr<Node> out_node;

    if (out_rg) {
        auto self_storage  = storage_;
        auto other_storage = other.storage_;
        auto self_nd       = input_node(*this);
        auto other_nd      = input_node(other);
        auto self_shape    = shape_;
        auto other_shape   = other.shape_;
        // Capture versions at forward time.
        int64_t self_ver   = storage_->version;
        int64_t other_ver  = other.storage_->version;

        out_node = std::make_shared<Node>();
        if (self_nd)  out_node->parents.push_back(self_nd);
        if (other_nd) out_node->parents.push_back(other_nd);

        out_node->backward_fn = [self_nd, other_nd, self_storage, other_storage, backend,
                                  self_shape, other_shape,
                                  M, K, N, self_ver, other_ver](const Tensor& grad) mutable {
            // dA = grad @ B^T
            if (self_nd) {
                check_version(other_storage, other_ver, "matmul", "rhs");
                Tensor g_a(self_shape);
                std::vector<double> b_transposed(static_cast<size_t>(N * K));
                for (int64_t k = 0; k < K; ++k) {
                    for (int64_t n = 0; n < N; ++n) {
                        b_transposed[static_cast<size_t>(n * K + k)] =
                            other_storage->data[static_cast<size_t>(k * N + n)];
                    }
                }
                backend->matmul(grad.data().data(), b_transposed.data(),
                                g_a.storage_->data.data(), M, N, K);
                distribute_grad(self_nd, g_a);
            }
            // dB = A^T @ grad
            if (other_nd) {
                check_version(self_storage, self_ver, "matmul", "lhs");
                Tensor g_b(other_shape);
                std::vector<double> a_transposed(static_cast<size_t>(K * M));
                for (int64_t m = 0; m < M; ++m) {
                    for (int64_t k = 0; k < K; ++k) {
                        a_transposed[static_cast<size_t>(k * M + m)] =
                            self_storage->data[static_cast<size_t>(m * K + k)];
                    }
                }
                backend->matmul(a_transposed.data(), grad.data().data(),
                                g_b.storage_->data.data(), K, M, N);
                distribute_grad(other_nd, g_b);
            }
        };
    }

    return make_output(std::move(out_storage), {M, N}, out_rg, std::move(out_node));
}

// ---------------------------------------------------------------------------
// transpose (2-D only)
//
// Returns a new Tensor with shape [N,M] where element [i,j] of the output
// equals element [j,i] of the input. Full autograd: the backward closure
// transposes the incoming gradient back to the original shape.
//
// Design decision: we store W as [out_features, in_features] and transpose
// at forward time to produce W^T:[in_features, out_features] for matmul.
// This is cleaner than pre-transposing W because matmul's backward operates
// on the stored (non-transposed) W values correctly.
// ---------------------------------------------------------------------------

Tensor Tensor::transpose() const {
    if (ndim() != 2) {
        throw std::invalid_argument(
            "Tensor::transpose: only 2-D tensors are supported (got shape " +
            shape_str(shape_) + ").");
    }

    const int64_t M = shape_[0], N = shape_[1];
    auto out_storage = std::make_shared<Storage>(M * N);
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            out_storage->data[static_cast<size_t>(j * M + i)] =
                storage_->data[static_cast<size_t>(i * N + j)];
        }
    }

    bool out_rg = grad_mode_enabled() && rg_;
    std::shared_ptr<Node> out_node;

    if (out_rg) {
        auto self_nd   = input_node(*this);
        auto orig_shape = shape_;

        out_node = std::make_shared<Node>();
        if (self_nd) out_node->parents.push_back(self_nd);

        // Backward: transpose the gradient back to the original [M,N] shape.
        out_node->backward_fn = [self_nd, orig_shape, M, N](const Tensor& grad) mutable {
            // grad has shape [N,M]; we need [M,N] for the original tensor.
            Tensor g(orig_shape);
            for (int64_t i = 0; i < M; ++i) {
                for (int64_t j = 0; j < N; ++j) {
                    g[i * N + j] = grad[j * M + i];
                }
            }
            distribute_grad(self_nd, g);
        };
    }

    return make_output(std::move(out_storage), {N, M}, out_rg, std::move(out_node));
}

// ---------------------------------------------------------------------------
// relu
//
// VERSION GUARD: backward closure reads self_storage to check sign of inputs.
// ---------------------------------------------------------------------------

Tensor Tensor::relu() const {
    auto out_storage = std::make_shared<Storage>(numel());
    for (int64_t i = 0; i < numel(); ++i) {
        const double x = storage_->data[static_cast<size_t>(i)];
        out_storage->data[static_cast<size_t>(i)] = x > 0.0 ? x : 0.0;
    }

    bool out_rg = grad_mode_enabled() && rg_;
    std::shared_ptr<Node> out_node;

    if (out_rg) {
        auto self_storage = storage_;
        auto self_nd      = input_node(*this);
        auto sh           = shape_;
        int64_t n         = numel();
        int64_t self_ver  = storage_->version;

        out_node = std::make_shared<Node>();
        if (self_nd) out_node->parents.push_back(self_nd);

        out_node->backward_fn = [self_nd, self_storage, sh, n,
                                  self_ver](const Tensor& grad) mutable {
            check_version(self_storage, self_ver, "relu", "input");
            Tensor g(sh);
            for (int64_t i = 0; i < n; ++i) {
                g[i] = (self_storage->data[static_cast<size_t>(i)] > 0.0) ? grad[i] : 0.0;
            }
            distribute_grad(self_nd, g);
        };
    }

    return make_output(std::move(out_storage), shape_, out_rg, std::move(out_node));
}

// ---------------------------------------------------------------------------
// square
//
// VERSION GUARD: backward closure reads self_storage for the 2x value.
// ---------------------------------------------------------------------------

Tensor Tensor::square() const {
    auto out_storage = std::make_shared<Storage>(numel());
    for (int64_t i = 0; i < numel(); ++i) {
        const double x = storage_->data[static_cast<size_t>(i)];
        out_storage->data[static_cast<size_t>(i)] = x * x;
    }

    bool out_rg = grad_mode_enabled() && rg_;
    std::shared_ptr<Node> out_node;

    if (out_rg) {
        auto self_storage = storage_;
        auto self_nd      = input_node(*this);
        auto sh           = shape_;
        int64_t n         = numel();
        int64_t self_ver  = storage_->version;

        out_node = std::make_shared<Node>();
        if (self_nd) out_node->parents.push_back(self_nd);

        out_node->backward_fn = [self_nd, self_storage, sh, n,
                                  self_ver](const Tensor& grad) mutable {
            check_version(self_storage, self_ver, "square", "input");
            Tensor g(sh);
            for (int64_t i = 0; i < n; ++i) {
                g[i] = 2.0 * self_storage->data[static_cast<size_t>(i)] * grad[i];
            }
            distribute_grad(self_nd, g);
        };
    }

    return make_output(std::move(out_storage), shape_, out_rg, std::move(out_node));
}

// ---------------------------------------------------------------------------
// mean → scalar
//
// No version guard needed: the backward closure only uses shape and numel,
// not actual input tensor values.
// ---------------------------------------------------------------------------

// PPO support operations.
Tensor Tensor::exp() const {
    auto out_storage = std::make_shared<Storage>(numel());
    for (int64_t i = 0; i < numel(); ++i) {
        out_storage->data[static_cast<size_t>(i)] =
            std::exp(storage_->data[static_cast<size_t>(i)]);
    }

    const bool out_rg = grad_mode_enabled() && rg_;
    std::shared_ptr<Node> out_node;
    if (out_rg) {
        auto self_nd = input_node(*this);
        auto self_storage = storage_;
        auto output_values = out_storage->data;
        auto sh = shape_;
        const int64_t n = numel();
        const int64_t self_ver = storage_->version;

        out_node = std::make_shared<Node>();
        if (self_nd) out_node->parents.push_back(self_nd);
        out_node->backward_fn = [self_nd, self_storage, output_values, sh, n,
                                 self_ver](const Tensor& grad) mutable {
            check_version(self_storage, self_ver, "exp", "input");
            Tensor g(sh);
            for (int64_t i = 0; i < n; ++i) {
                g[i] = grad[i] * output_values[static_cast<size_t>(i)];
            }
            distribute_grad(self_nd, g);
        };
    }
    return make_output(std::move(out_storage), shape_, out_rg, std::move(out_node));
}

Tensor Tensor::log() const {
    auto out_storage = std::make_shared<Storage>(numel());
    for (int64_t i = 0; i < numel(); ++i) {
        const double value = storage_->data[static_cast<size_t>(i)];
        if (!(value > 0.0)) {
            throw std::domain_error("Tensor::log requires strictly positive values");
        }
        out_storage->data[static_cast<size_t>(i)] = std::log(value);
    }

    const bool out_rg = grad_mode_enabled() && rg_;
    std::shared_ptr<Node> out_node;
    if (out_rg) {
        auto self_nd = input_node(*this);
        auto self_storage = storage_;
        auto sh = shape_;
        const int64_t n = numel();
        const int64_t self_ver = storage_->version;

        out_node = std::make_shared<Node>();
        if (self_nd) out_node->parents.push_back(self_nd);
        out_node->backward_fn = [self_nd, self_storage, sh, n,
                                 self_ver](const Tensor& grad) mutable {
            check_version(self_storage, self_ver, "log", "input");
            Tensor g(sh);
            for (int64_t i = 0; i < n; ++i) {
                g[i] = grad[i] / self_storage->data[static_cast<size_t>(i)];
            }
            distribute_grad(self_nd, g);
        };
    }
    return make_output(std::move(out_storage), shape_, out_rg, std::move(out_node));
}

Tensor Tensor::clamp(double min_value, double max_value) const {
    if (std::isnan(min_value) || std::isnan(max_value) ||
        min_value > max_value) {
        throw std::invalid_argument("Tensor::clamp requires min_value <= max_value");
    }
    auto out_storage = std::make_shared<Storage>(numel());
    for (int64_t i = 0; i < numel(); ++i) {
        out_storage->data[static_cast<size_t>(i)] =
            std::clamp(storage_->data[static_cast<size_t>(i)], min_value, max_value);
    }

    const bool out_rg = grad_mode_enabled() && rg_;
    std::shared_ptr<Node> out_node;
    if (out_rg) {
        auto self_nd = input_node(*this);
        auto self_storage = storage_;
        auto sh = shape_;
        const int64_t n = numel();
        const int64_t self_ver = storage_->version;

        out_node = std::make_shared<Node>();
        if (self_nd) out_node->parents.push_back(self_nd);
        out_node->backward_fn = [self_nd, self_storage, sh, n, min_value,
                                 max_value, self_ver](const Tensor& grad) mutable {
            check_version(self_storage, self_ver, "clamp", "input");
            Tensor g(sh);
            for (int64_t i = 0; i < n; ++i) {
                const double value = self_storage->data[static_cast<size_t>(i)];
                g[i] = (value > min_value && value < max_value) ? grad[i] : 0.0;
            }
            distribute_grad(self_nd, g);
        };
    }
    return make_output(std::move(out_storage), shape_, out_rg, std::move(out_node));
}

Tensor Tensor::minimum(const Tensor& other) const {
    check_same_shape(*this, other, "Tensor::minimum");
    auto out_storage = std::make_shared<Storage>(numel());
    std::vector<uint8_t> choose_lhs(static_cast<size_t>(numel()));
    for (int64_t i = 0; i < numel(); ++i) {
        const bool lhs = storage_->data[static_cast<size_t>(i)] <=
                         other.storage_->data[static_cast<size_t>(i)];
        choose_lhs[static_cast<size_t>(i)] = lhs ? 1 : 0;
        out_storage->data[static_cast<size_t>(i)] =
            lhs ? storage_->data[static_cast<size_t>(i)]
                : other.storage_->data[static_cast<size_t>(i)];
    }

    const bool out_rg = grad_mode_enabled() && (rg_ || other.rg_);
    std::shared_ptr<Node> out_node;
    if (out_rg) {
        auto self_nd = input_node(*this);
        auto other_nd = input_node(other);
        auto self_storage = storage_;
        auto other_storage = other.storage_;
        auto sh = shape_;
        const int64_t n = numel();
        const int64_t self_ver = storage_->version;
        const int64_t other_ver = other.storage_->version;

        out_node = std::make_shared<Node>();
        if (self_nd) out_node->parents.push_back(self_nd);
        if (other_nd) out_node->parents.push_back(other_nd);
        out_node->backward_fn = [self_nd, other_nd, self_storage, other_storage,
                                 choose_lhs, sh, n, self_ver,
                                 other_ver](const Tensor& grad) mutable {
            check_version(self_storage, self_ver, "minimum", "lhs");
            check_version(other_storage, other_ver, "minimum", "rhs");
            if (self_nd) {
                Tensor g(sh);
                for (int64_t i = 0; i < n; ++i) {
                    if (choose_lhs[static_cast<size_t>(i)]) g[i] = grad[i];
                }
                distribute_grad(self_nd, g);
            }
            if (other_nd) {
                Tensor g(sh);
                for (int64_t i = 0; i < n; ++i) {
                    if (!choose_lhs[static_cast<size_t>(i)]) g[i] = grad[i];
                }
                distribute_grad(other_nd, g);
            }
        };
    }
    return make_output(std::move(out_storage), shape_, out_rg, std::move(out_node));
}

Tensor Tensor::log_softmax() const {
    if (ndim() != 2 || shape_[0] <= 0 || shape_[1] <= 0) {
        throw std::invalid_argument(
            "Tensor::log_softmax requires a non-empty 2-D [B,C] tensor");
    }
    const int64_t B = shape_[0];
    const int64_t C = shape_[1];
    auto out_storage = std::make_shared<Storage>(numel());
    std::vector<double> probabilities(static_cast<size_t>(numel()));

    for (int64_t row = 0; row < B; ++row) {
        double max_value = storage_->data[static_cast<size_t>(row * C)];
        for (int64_t col = 1; col < C; ++col) {
            max_value = std::max(
                max_value, storage_->data[static_cast<size_t>(row * C + col)]);
        }
        double exp_sum = 0.0;
        for (int64_t col = 0; col < C; ++col) {
            exp_sum += std::exp(
                storage_->data[static_cast<size_t>(row * C + col)] - max_value);
        }
        const double log_sum_exp = max_value + std::log(exp_sum);
        for (int64_t col = 0; col < C; ++col) {
            const size_t index = static_cast<size_t>(row * C + col);
            out_storage->data[index] = storage_->data[index] - log_sum_exp;
            probabilities[index] = std::exp(out_storage->data[index]);
        }
    }

    const bool out_rg = grad_mode_enabled() && rg_;
    std::shared_ptr<Node> out_node;
    if (out_rg) {
        auto self_nd = input_node(*this);
        auto self_storage = storage_;
        auto sh = shape_;
        const int64_t self_ver = storage_->version;

        out_node = std::make_shared<Node>();
        if (self_nd) out_node->parents.push_back(self_nd);
        out_node->backward_fn = [self_nd, self_storage, probabilities, sh, B, C,
                                 self_ver](const Tensor& grad) mutable {
            check_version(self_storage, self_ver, "log_softmax", "input");
            Tensor g(sh);
            for (int64_t row = 0; row < B; ++row) {
                double row_sum = 0.0;
                for (int64_t col = 0; col < C; ++col) {
                    row_sum += grad[row * C + col];
                }
                for (int64_t col = 0; col < C; ++col) {
                    const size_t index = static_cast<size_t>(row * C + col);
                    g[row * C + col] = grad[row * C + col] -
                                       probabilities[index] * row_sum;
                }
            }
            distribute_grad(self_nd, g);
        };
    }
    return make_output(std::move(out_storage), shape_, out_rg, std::move(out_node));
}

// Mean over all elements to a scalar. Its backward closure only needs shape.
Tensor Tensor::mean() const {
    if (numel() == 0) {
        throw std::invalid_argument("Tensor::mean requires a non-empty tensor");
    }
    const double count_d = static_cast<double>(numel());
    double sum = 0.0;
    for (int64_t i = 0; i < numel(); ++i) {
        sum += storage_->data[static_cast<size_t>(i)];
    }

    auto out_storage = std::make_shared<Storage>(std::vector<double>{sum / count_d});

    bool out_rg = grad_mode_enabled() && rg_;
    std::shared_ptr<Node> out_node;

    if (out_rg) {
        auto self_nd = input_node(*this);
        auto sh      = shape_;
        int64_t n    = numel();

        out_node = std::make_shared<Node>();
        if (self_nd) out_node->parents.push_back(self_nd);

        out_node->backward_fn = [self_nd, sh, n](const Tensor& grad) mutable {
            const double upstream = grad[0];
            Tensor g(sh);
            const double val = upstream / static_cast<double>(n);
            for (int64_t i = 0; i < n; ++i) {
                g[i] = val;
            }
            distribute_grad(self_nd, g);
        };
    }

    return make_output(std::move(out_storage), {}, out_rg, std::move(out_node));
}

// ---------------------------------------------------------------------------
// gather
//
// No version guard needed: the backward closure only uses idx_copy (a copy
// of the integer column indices), not the actual input storage values.
// ---------------------------------------------------------------------------

Tensor Tensor::gather(const Tensor& indices) const {
    if (ndim() != 2) {
        throw std::invalid_argument(
            "Tensor::gather: 'this' must be 2-D (got shape " + shape_str(shape_) + ").");
    }
    if (indices.ndim() != 1) {
        throw std::invalid_argument(
            "Tensor::gather: 'indices' must be 1-D (got shape " +
            shape_str(indices.shape_) + ").");
    }

    const int64_t N = shape_[0], C = shape_[1];
    if (N <= 0 || C <= 0) {
        throw std::invalid_argument(
            "Tensor::gather requires a non-empty 2-D source tensor");
    }
    if (indices.numel() != N) {
        throw std::invalid_argument(
            "Tensor::gather: indices length (" + std::to_string(indices.numel()) +
            ") must equal number of rows (" + std::to_string(N) + ").");
    }

    auto out_storage = std::make_shared<Storage>(N);
    std::vector<int64_t> idx_copy;
    idx_copy.reserve(static_cast<size_t>(N));

    for (int64_t i = 0; i < N; ++i) {
        const double raw_index = indices[i];
        if (!std::isfinite(raw_index) || std::trunc(raw_index) != raw_index ||
            raw_index < 0.0 || raw_index >= static_cast<double>(C)) {
            throw std::invalid_argument(
                "Tensor::gather: indices must contain in-range integer values");
        }
        const int64_t col = static_cast<int64_t>(raw_index);
        if (col < 0 || col >= C) {
            throw std::invalid_argument(
                "Tensor::gather: index " + std::to_string(col) +
                " at position " + std::to_string(i) +
                " is out of bounds for dim size " + std::to_string(C) + ".");
        }
        out_storage->data[static_cast<size_t>(i)] =
            storage_->data[static_cast<size_t>(i * C + col)];
        idx_copy.push_back(col);
    }

    bool out_rg = grad_mode_enabled() && rg_;
    std::shared_ptr<Node> out_node;

    if (out_rg) {
        auto self_nd = input_node(*this);
        auto sh      = shape_;

        out_node = std::make_shared<Node>();
        if (self_nd) out_node->parents.push_back(self_nd);

        out_node->backward_fn = [self_nd, sh, idx_copy, N, C](const Tensor& grad) mutable {
            // Scatter-add: grad_input[i, idx_copy[i]] += grad[i]
            Tensor g(sh);
            for (int64_t i = 0; i < N; ++i) {
                const int64_t col = idx_copy[static_cast<size_t>(i)];
                g[i * C + col] += grad[i];
            }
            distribute_grad(self_nd, g);
        };
    }

    return make_output(std::move(out_storage), {N}, out_rg, std::move(out_node));
}

// ---------------------------------------------------------------------------
// max_last_dim — row-wise maximum: [B, C] -> [B].
//
// Forward: for each row i, find the maximum value across columns.
//   Tie-breaking rule: first occurrence (lowest column index) wins.
//   This is deterministic and consistent with np.argmax.
//
// Backward: one-hot gradient routing to the argmax position.
//   grad_input[i, argmax_i] += upstream_grad[i]
//   grad_input[i, j]        += 0  for all j != argmax_i
//
// VERSION GUARD: the backward closure stores argmax_indices, which are derived
// from the input values at forward time. If the input is mutated in-place after
// forward (bumping its version), those indices would be stale. The version check
// detects this, consistent with mul/matmul/square/relu (Milestone 6 rule).
// ---------------------------------------------------------------------------

Tensor Tensor::max_last_dim() const {
    if (ndim() != 2 || shape_[0] <= 0 || shape_[1] <= 0) {
        throw std::invalid_argument(
            "Tensor::max_last_dim: tensor must be exactly 2-D [B, C], got shape " +
            shape_str(shape_) + ". Only the [B, C] -> [B] row-wise max is supported.");
    }

    const int64_t B = shape_[0];
    const int64_t C = shape_[1];

    auto out_storage = std::make_shared<Storage>(B);
    std::vector<int64_t> argmax_indices(static_cast<size_t>(B));

    for (int64_t i = 0; i < B; ++i) {
        double max_val = storage_->data[static_cast<size_t>(i * C)];
        int64_t max_col = 0;
        // First-occurrence tie-breaking: strictly > keeps the first maximum found.
        for (int64_t j = 1; j < C; ++j) {
            const double val = storage_->data[static_cast<size_t>(i * C + j)];
            if (val > max_val) {
                max_val = val;
                max_col = j;
            }
        }
        out_storage->data[static_cast<size_t>(i)] = max_val;
        argmax_indices[static_cast<size_t>(i)] = max_col;
    }

    bool out_rg = grad_mode_enabled() && rg_;
    std::shared_ptr<Node> out_node;

    if (out_rg) {
        auto self_nd      = input_node(*this);
        auto self_storage = storage_;
        auto sh           = shape_;
        int64_t self_ver  = storage_->version;

        out_node = std::make_shared<Node>();
        if (self_nd) out_node->parents.push_back(self_nd);

        out_node->backward_fn = [self_nd, self_storage, sh, argmax_indices, B, C,
                                  self_ver](const Tensor& grad) mutable {
            check_version(self_storage, self_ver, "max_last_dim", "input");
            // One-hot routing: gradient flows only to the argmax column per row.
            Tensor g(sh);
            for (int64_t i = 0; i < B; ++i) {
                const int64_t col = argmax_indices[static_cast<size_t>(i)];
                g[i * C + col] += grad[i];
            }
            distribute_grad(self_nd, g);
        };
    }

    return make_output(std::move(out_storage), {B}, out_rg, std::move(out_node));
}

} // namespace rl::tensor
