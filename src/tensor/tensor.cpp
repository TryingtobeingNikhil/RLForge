#include "rl/tensor/tensor.hpp"

#include <algorithm>
#include <cmath>
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
    for (int64_t d : shape) { n *= d; }
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
            ". Tensor-tensor broadcasting is not supported in this milestone.");
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
// `parent_node` is always non-null here (every input to a differentiable op
// that requires grad has a Node — see make_leaf_node below). If the input
// was a leaf tensor created by the user, we gave it a leaf Node whose
// backward_fn accumulates into its grad buffer. If it was an intermediate
// result, its Node's backward_fn chains further.
//
// This avoids storing raw Tensor* pointers in closures (which would dangle
// when the Tensor is passed by value into a lambda or loses scope).
// ---------------------------------------------------------------------------
void distribute_grad(std::shared_ptr<Node>& parent_node, const Tensor& grad_contribution) {
    if (!parent_node) return;
    if (!parent_node->incoming_grad) {
        parent_node->incoming_grad =
            std::make_shared<Tensor>(grad_contribution.shape());
    }
    auto& dst = parent_node->incoming_grad->data_mutable();
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
    // This ensures repeated backward() calls (without zero_grad) only
    // accumulate through the leaf grad_ buffer (intentional), not through
    // stale intermediate node buffers from a previous backward pass.
    for (auto& node : order) {
        node->incoming_grad = nullptr;
    }

    root_node->incoming_grad = std::make_shared<Tensor>(std::move(initial_grad));

    for (auto& node : order) {
        if (!node->incoming_grad || !node->backward_fn) {
            continue;
        }
        // CORRECTION #2: run every backward_fn under no_grad().
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
// the grad buffer (shared_ptr<Tensor> for the grad, or the Storage directly)
// avoids the dangling-pointer problem that would occur if we stored a raw
// Tensor* pointer (which would dangle if the leaf Tensor is passed by value).
//
// We store a shared_ptr<Storage> for the grad storage, allocating it on first
// use. The leaf Node's incoming_grad IS the leaf's grad buffer.
//
// To do this cleanly: the leaf node's backward_fn reads incoming_grad directly
// and accumulates it into the leaf's grad_ buffer (shared_ptr<Tensor>).
// We capture that grad_ shared_ptr by value.
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
        auto& dst = grad_ptr->data_mutable();
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
//
// If the operand requires grad and has a node, return its node (which will
// become a parent of the output node). If requires_grad is true but it
// has no node yet (shouldn't happen after requires_grad_(true) now creates
// a leaf node), we skip. If requires_grad is false, return nullptr.
// ---------------------------------------------------------------------------
namespace {

inline std::shared_ptr<Node> input_node(const Tensor& t) {
    return t.requires_grad() ? t.node() : nullptr;
}

} // namespace

// ---------------------------------------------------------------------------
// add
// ---------------------------------------------------------------------------

Tensor Tensor::add(const Tensor& other) const {
    check_same_shape(*this, other, "Tensor::add");

    auto out_storage = std::make_shared<Storage>(numel());
    for (int64_t i = 0; i < numel(); ++i) {
        out_storage->data[static_cast<size_t>(i)] =
            storage_->data[static_cast<size_t>(i)] +
            other.storage_->data[static_cast<size_t>(i)];
    }

    bool out_rg = grad_mode_enabled() && (rg_ || other.rg_);
    std::shared_ptr<Node> out_node;

    if (out_rg) {
        auto self_nd  = input_node(*this);
        auto other_nd = input_node(other);

        out_node = std::make_shared<Node>();
        if (self_nd)  out_node->parents.push_back(self_nd);
        if (other_nd) out_node->parents.push_back(other_nd);

        out_node->backward_fn = [self_nd, other_nd](const Tensor& grad) mutable {
            // d(a+b)/da = 1, d(a+b)/db = 1
            if (self_nd)  distribute_grad(self_nd, grad);
            if (other_nd) distribute_grad(other_nd, grad);
        };
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

        out_node = std::make_shared<Node>();
        if (self_nd)  out_node->parents.push_back(self_nd);
        if (other_nd) out_node->parents.push_back(other_nd);

        out_node->backward_fn = [self_nd, other_nd, self_storage, other_storage,
                                  sh, n](const Tensor& grad) mutable {
            if (self_nd) {
                Tensor g(sh);
                for (int64_t i = 0; i < n; ++i) {
                    g[i] = grad[i] * other_storage->data[static_cast<size_t>(i)];
                }
                distribute_grad(self_nd, g);
            }
            if (other_nd) {
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

    if (K != K2) {
        throw std::invalid_argument(
            "Tensor::matmul: inner dimension mismatch: lhs shape " + shape_str(shape_) +
            " has K=" + std::to_string(K) + " but rhs shape " +
            shape_str(other.shape_) + " has rows=" + std::to_string(K2) + ".");
    }

    auto out_storage = std::make_shared<Storage>(M * N);
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t nc = 0; nc < N; ++nc) {
            double acc = 0.0;
            for (int64_t k = 0; k < K; ++k) {
                acc += storage_->data[static_cast<size_t>(m * K + k)] *
                       other.storage_->data[static_cast<size_t>(k * N + nc)];
            }
            out_storage->data[static_cast<size_t>(m * N + nc)] = acc;
        }
    }

    bool out_rg = grad_mode_enabled() && (rg_ || other.rg_);
    std::shared_ptr<Node> out_node;

    if (out_rg) {
        auto self_storage  = storage_;
        auto other_storage = other.storage_;
        auto self_nd       = input_node(*this);
        auto other_nd      = input_node(other);
        auto self_shape    = shape_;
        auto other_shape   = other.shape_;

        out_node = std::make_shared<Node>();
        if (self_nd)  out_node->parents.push_back(self_nd);
        if (other_nd) out_node->parents.push_back(other_nd);

        out_node->backward_fn = [self_nd, other_nd, self_storage, other_storage,
                                  self_shape, other_shape,
                                  M, K, N](const Tensor& grad) mutable {
            // dA = grad @ B^T
            if (self_nd) {
                Tensor g_a(self_shape);
                for (int64_t m = 0; m < M; ++m) {
                    for (int64_t k = 0; k < K; ++k) {
                        double acc = 0.0;
                        for (int64_t nc = 0; nc < N; ++nc) {
                            acc += grad[m * N + nc] *
                                   other_storage->data[static_cast<size_t>(k * N + nc)];
                        }
                        g_a[m * K + k] = acc;
                    }
                }
                distribute_grad(self_nd, g_a);
            }
            // dB = A^T @ grad
            if (other_nd) {
                Tensor g_b(other_shape);
                for (int64_t k = 0; k < K; ++k) {
                    for (int64_t nc = 0; nc < N; ++nc) {
                        double acc = 0.0;
                        for (int64_t m = 0; m < M; ++m) {
                            acc += self_storage->data[static_cast<size_t>(m * K + k)] *
                                   grad[m * N + nc];
                        }
                        g_b[k * N + nc] = acc;
                    }
                }
                distribute_grad(other_nd, g_b);
            }
        };
    }

    return make_output(std::move(out_storage), {M, N}, out_rg, std::move(out_node));
}

// ---------------------------------------------------------------------------
// relu
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

        out_node = std::make_shared<Node>();
        if (self_nd) out_node->parents.push_back(self_nd);

        out_node->backward_fn = [self_nd, self_storage, sh, n](const Tensor& grad) mutable {
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

        out_node = std::make_shared<Node>();
        if (self_nd) out_node->parents.push_back(self_nd);

        out_node->backward_fn = [self_nd, self_storage, sh, n](const Tensor& grad) mutable {
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
// ---------------------------------------------------------------------------

Tensor Tensor::mean() const {
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
    if (indices.numel() != N) {
        throw std::invalid_argument(
            "Tensor::gather: indices length (" + std::to_string(indices.numel()) +
            ") must equal number of rows (" + std::to_string(N) + ").");
    }

    auto out_storage = std::make_shared<Storage>(N);
    std::vector<int64_t> idx_copy;
    idx_copy.reserve(static_cast<size_t>(N));

    for (int64_t i = 0; i < N; ++i) {
        const int64_t col = static_cast<int64_t>(indices[i]);
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
            // Accumulates correctly even when duplicate indices are present.
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

} // namespace rl::tensor
