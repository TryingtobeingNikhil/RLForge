#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "rl/tensor/autograd.hpp"

namespace rl::tensor {

// ---------------------------------------------------------------------------
// Storage — the contiguous backing buffer for a Tensor.
//
// VERSION COUNTER (Milestone 6):
//   `version` is incremented by any in-place mutation of the data buffer.
//   Backward closures record the Storage version at closure-creation time
//   (i.e. at forward-pass time) and check it at backward time before using
//   captured tensor values for gradient computation. A mismatch throws
//   std::runtime_error identifying the stale operation.
//
//   What this catches: in-place mutation of a tensor AFTER it was used in a
//   forward pass that built an autograd graph, before backward() is called.
//   What this does NOT catch: mutations before the forward pass (correct
//   behaviour — version is captured at closure-creation time, not object-
//   creation time), or mutations to tensors not participating in any graph.
// ---------------------------------------------------------------------------
struct Storage {
    explicit Storage(int64_t size)
        : data(checked_size(size), 0.0), size(size) {}
    explicit Storage(std::vector<double> buf)
        : data(std::move(buf)), size(static_cast<int64_t>(data.size())) {}

    std::vector<double> data;
    int64_t size;

    // Incremented by every in-place mutation. Read by backward closures.
    int64_t version = 0;

    void bump_version() noexcept { ++version; }

private:
    static size_t checked_size(int64_t value) {
        if (value < 0) {
            throw std::invalid_argument("Tensor storage size must be non-negative");
        }
        return static_cast<size_t>(value);
    }
};

// A tracked mutable view over Tensor storage. It intentionally exposes no raw
// pointer, iterator, or std::vector reference: every mutation passes through a
// proxy that increments the Storage version at the moment of mutation. This
// keeps stale-graph detection correct even when the view outlives the call to
// Tensor::data_mutable().
class MutableTensorData {
public:
    class Reference {
    public:
        Reference(std::shared_ptr<Storage> storage, size_t index)
            : storage_(std::move(storage)), index_(index) {}

        operator double() const noexcept { return storage_->data[index_]; }

        Reference& operator=(double value) noexcept {
            storage_->bump_version();
            storage_->data[index_] = value;
            return *this;
        }
        Reference& operator=(const Reference& other) noexcept {
            return *this = static_cast<double>(other);
        }
        Reference& operator+=(double value) noexcept {
            return *this = static_cast<double>(*this) + value;
        }
        Reference& operator-=(double value) noexcept {
            return *this = static_cast<double>(*this) - value;
        }
        Reference& operator*=(double value) noexcept {
            return *this = static_cast<double>(*this) * value;
        }

    private:
        std::shared_ptr<Storage> storage_;
        size_t index_;
    };

    explicit MutableTensorData(std::shared_ptr<Storage> storage)
        : storage_(std::move(storage)) {}

    MutableTensorData(const MutableTensorData&) = default;
    MutableTensorData& operator=(const MutableTensorData&) = delete;

    size_t size() const noexcept { return storage_->data.size(); }
    Reference operator[](int64_t index) {
        return Reference(storage_, checked_index(index));
    }
    double operator[](int64_t index) const {
        return storage_->data[checked_index(index)];
    }

    MutableTensorData& operator=(const std::vector<double>& values) {
        if (values.size() != storage_->data.size()) {
            throw std::invalid_argument(
                "Mutable tensor data assignment must preserve the storage size");
        }
        auto replacement = values;
        storage_->bump_version();
        storage_->data = std::move(replacement);
        return *this;
    }

private:
    size_t checked_index(int64_t index) const {
        if (index < 0 || static_cast<size_t>(index) >= storage_->data.size()) {
            throw std::out_of_range("Tensor data index out of range");
        }
        return static_cast<size_t>(index);
    }

    std::shared_ptr<Storage> storage_;
};

// ---------------------------------------------------------------------------
// Tensor — dense, contiguous, row-major N-D array of doubles with optional
//          reverse-mode autograd support.
//
// Scalar type is fixed as double throughout this library (no template param).
// Shape is a general std::vector<int64_t>. This library correctly supports
// 0-D (scalar), 1-D, and 2-D tensors; ops that require a specific rank
// (e.g. matmul requires exactly 2-D) throw std::invalid_argument otherwise.
//
// IN-PLACE MUTATION SAFETY (Milestone 6):
//   Every assignment through data_mutable() increments the Storage version
//   counter. Backward closures that read input tensor values at backward time
//   record the Storage version at closure-creation time and throw
//   std::runtime_error on mismatch.
//   See Storage::version for the full semantics.
//
// BROADCASTING (Milestone 6 — scoped):
//   Tensor::add supports two shape combinations:
//     (a) Exact shape match — same as before.
//     (b) [B,N] + [N] — row-wise broadcast of the [N] bias across batch B.
//   Only this one pattern is supported. All other combinations throw
//   std::invalid_argument. General N-D NumPy-style broadcasting is NOT
//   implemented.
//
// KNOWN LIMITATION — VIEWS / SLICING / STRIDED TENSORS:
//   Not supported. Every Tensor owns its own contiguous storage.
// ---------------------------------------------------------------------------
class Tensor {
public:
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    // Construct a zero-filled tensor of the given shape.
    explicit Tensor(std::vector<int64_t> shape);

    // Construct from an existing data vector. Total elements must equal the
    // product of shape dimensions, or std::invalid_argument is thrown.
    Tensor(std::vector<double> data, std::vector<int64_t> shape);

    // -----------------------------------------------------------------------
    // Static factories
    // -----------------------------------------------------------------------
    static Tensor zeros(std::vector<int64_t> shape);
    static Tensor ones(std::vector<int64_t> shape);
    static Tensor from_data(std::vector<double> data, std::vector<int64_t> shape);

    // -----------------------------------------------------------------------
    // Shape / size
    // -----------------------------------------------------------------------
    const std::vector<int64_t>& shape() const noexcept { return shape_; }
    int64_t ndim() const noexcept { return static_cast<int64_t>(shape_.size()); }
    int64_t numel() const noexcept { return storage_->size; }

    // Returns the scalar value; throws std::invalid_argument if numel() != 1.
    double item() const;

    // -----------------------------------------------------------------------
    // Data access
    // -----------------------------------------------------------------------
    const std::vector<double>& data() const noexcept { return storage_->data; }

    // Returns a tracked mutable view. Each indexed or whole-buffer assignment
    // increments the Storage version counter at mutation time.
    MutableTensorData data_mutable() noexcept { return MutableTensorData(storage_); }

    // The current version of this tensor's backing Storage. Backward closures
    // capture this at forward-pass time and compare at backward time.
    int64_t storage_version() const noexcept { return storage_->version; }

    double operator[](int64_t i) const;
    MutableTensorData::Reference operator[](int64_t i) {
        return data_mutable()[i];
    }

    // -----------------------------------------------------------------------
    // Autograd
    // -----------------------------------------------------------------------

    // Enable/disable gradient tracking. Returns *this for chaining:
    //   auto x = Tensor::zeros({3}).requires_grad_(true);
    // Note: the method is named with a trailing underscore (requires_grad_)
    // matching PyTorch convention; the private data member uses rg_ to avoid
    // a name collision.
    Tensor& requires_grad_(bool rg);
    bool requires_grad() const noexcept { return rg_; }

    // Accumulated gradient (only meaningful on leaf tensors with
    // requires_grad()==true). Returns nullptr if backward() has not run yet
    // or after zero_grad().
    const Tensor* grad() const noexcept { return grad_.get(); }

    // Accumulate `g` into this tensor's grad buffer (+=). Allocates the
    // buffer on first call. Valid on leaf tensors with requires_grad()==true.
    void accumulate_grad(const Tensor& g);

    // Zero out the gradient buffer (does not free it; does not auto-zero
    // on backward — you must call this explicitly).
    void zero_grad();

    // -----------------------------------------------------------------------
    // Graph node (used internally by ops and backward())
    // -----------------------------------------------------------------------
    const std::shared_ptr<Node>& node() const noexcept { return node_; }
    void set_node(std::shared_ptr<Node> n) { node_ = std::move(n); }

    // -----------------------------------------------------------------------
    // backward() — run reverse-mode automatic differentiation.
    //
    // backward() (no arg): only valid on scalar tensors (numel()==1).
    //   Seeds the backward pass with grad=1.
    //
    // backward(Tensor grad): for non-scalar outputs, accepts an explicit
    //   upstream gradient of the same shape as this tensor.
    //
    // Algorithm:
    //   1. Build topological order via iterative post-order DFS.
    //   2. Seed this node's incoming_grad buffer with the initial gradient.
    //   3. Traverse in REVERSE topo order (output → leaves). For each node,
    //      call backward_fn(incoming_grad) ONLY AFTER all downstream consumers
    //      have already accumulated into incoming_grad. This guarantees
    //      diamond-dependency correctness.
    //   4. Each backward_fn runs under no_grad() to prevent building a second
    //      autograd graph on top of the backward pass.
    //   5. backward_fn distributes gradients to its input nodes' incoming_grad
    //      buffers (or directly to leaf tensors' grad buffers).
    //   6. Gradients accumulate (+=) — no auto-zero. Call zero_grad() explicitly.
    //
    // VERSION GUARD: backward closures that read saved input tensor values
    //   (mul, matmul, relu, square) check that the captured Storage version
    //   matches the current version. A mismatch throws std::runtime_error.
    // -----------------------------------------------------------------------
    void backward();
    void backward(Tensor grad);

    // -----------------------------------------------------------------------
    // detach() — new Tensor sharing storage, requires_grad=false, no Node.
    // -----------------------------------------------------------------------
    Tensor detach() const;

    // -----------------------------------------------------------------------
    // Forward ops — named methods; operator overloads are thin wrappers.
    // -----------------------------------------------------------------------

    // Elementwise addition.
    // Broadcasting: supports [B,N]+[N] (row-wise broadcast of [N] across batch B).
    // All other shape combinations require exact match or throw std::invalid_argument.
    Tensor add(const Tensor& other) const;

    Tensor sub(const Tensor& other) const;
    Tensor mul(const Tensor& other) const;  // elementwise
    Tensor mul(double scalar) const;        // scalar broadcast

    // 2-D × 2-D only. Shape [M,K] @ [K,N] → [M,N].
    Tensor matmul(const Tensor& other) const;

    // Transpose a 2-D tensor. Shape [M,N] → [N,M].
    // Full autograd: backward transposes the incoming gradient.
    // Throws std::invalid_argument for non-2-D tensors.
    Tensor transpose() const;

    // ReLU: max(0, x). Subgradient = 0 at x=0.
    Tensor relu() const;

    // Elementwise x^2.
    Tensor square() const;

    // Elementwise exponential and natural logarithm. log() requires every
    // input element to be strictly positive.
    Tensor exp() const;
    Tensor log() const;

    // Elementwise clamp to [min_value, max_value]. The backward derivative
    // is one strictly inside the interval and zero at/outside the bounds.
    Tensor clamp(double min_value, double max_value) const;

    // Elementwise minimum. Shapes must match; ties route gradient to lhs.
    Tensor minimum(const Tensor& other) const;

    // Numerically stable row-wise log-softmax for a [B,C] tensor.
    Tensor log_softmax() const;

    // Mean over all elements → scalar tensor (shape {}).
    Tensor mean() const;

    // Gather: `this` must be 2-D [N,C], `indices` must be 1-D [N].
    // Returns 1-D [N]: out[i] = this[i, indices[i]].
    // Backward is scatter-add: grad_input[i, indices[i]] += upstream_grad[i].
    // Duplicate indices accumulate (not overwrite) — see test coverage.
    Tensor gather(const Tensor& indices) const;

    // max_last_dim — row-wise maximum of a 2-D tensor: [B, C] -> [B].
    //
    // Returns the maximum value per row. Throws std::invalid_argument if the
    // tensor is not exactly 2-D.
    //
    // TIE-BREAKING RULE: first occurrence (lowest column index) wins when two
    // or more elements in the same row are equal. This is deterministic and
    // consistent with np.argmax. The rule is defined, tested, and stable.
    //
    // BACKWARD: gradient flows only to the argmax position for each row
    // (one-hot routing). For row i with argmax at column k:
    //   grad_input[i, k] += upstream_grad[i]
    //   grad_input[i, j] += 0  for all j != k
    // This is identical in spirit to gather's scatter-add backward.
    //
    // VERSION GUARD: the backward closure saves the argmax indices which are
    // a function of the input values at forward time. If the input Storage is
    // mutated in-place after forward (bumping its version), the saved indices
    // would be stale. The version guard detects this and throws, consistent
    // with how mul/matmul/square/relu are handled (Milestone 6 rule).
    //
    // DQN usage:
    //   next_q_max = target_net.forward(next_states).max_last_dim()  // [B]
    //   (called under no_grad(), so version guard never fires in practice)
    Tensor max_last_dim() const;

    // -----------------------------------------------------------------------
    // Operator overloads
    // -----------------------------------------------------------------------
    Tensor operator+(const Tensor& other) const { return add(other); }
    Tensor operator-(const Tensor& other) const { return sub(other); }
    Tensor operator*(const Tensor& other) const { return mul(other); }
    Tensor operator*(double scalar) const { return mul(scalar); }

private:
    std::shared_ptr<Storage> storage_;
    std::vector<int64_t> shape_;

    // Private member is rg_ to avoid a name clash with the public method
    // requires_grad_(bool) which follows the PyTorch trailing-underscore
    // convention for in-place setters.
    bool rg_ = false;

    // Gradient buffer: only allocated for leaf tensors with rg_=true.
    std::shared_ptr<Tensor> grad_;

    // Autograd graph node. Null for leaves or tensors created under no_grad().
    std::shared_ptr<Node> node_;

    // Private constructor used by op implementations.
    static Tensor make_output(std::shared_ptr<Storage> storage, std::vector<int64_t> shape,
                              bool rg, std::shared_ptr<Node> node);

    // Create a leaf Node for a tensor with requires_grad=true.
    // The node's backward_fn captures grad_ by shared_ptr so gradient
    // accumulation is safe even when the Tensor object is copied or
    // passed by value into closures.
    std::shared_ptr<Node> make_leaf_node();
};

// Free-function scalar-on-left operator: 3.0 * tensor
inline Tensor operator*(double scalar, const Tensor& t) { return t.mul(scalar); }

} // namespace rl::tensor
