#pragma once

#include <functional>
#include <memory>
#include <vector>

// Forward-declare Tensor so Node's backward_fn signature can reference it
// without including the full tensor header (which itself includes this header).
namespace rl::tensor {
class Tensor;
} // namespace rl::tensor

namespace rl::tensor {

// ---------------------------------------------------------------------------
// Node — one vertex in the dynamic computational graph.
//
// Ownership model (correction to initial plan):
//   Tensor  --shared_ptr--> Node  --shared_ptr--> parent Nodes
//
// Direction matters: the output tensor owns its node; each node owns the
// nodes of the tensors that were its inputs (its "parents" in the forward
// direction, i.e. the operands). There is NO back-edge from a Node to any
// Tensor that owns it — so there is NO reference cycle.
//
// Why NOT weak_ptr for parents? Because parent nodes need to stay alive
// for the duration of backward(). The only path that would create a cycle
// is Tensor→Node→Tensor (a node capturing an input Tensor's shared_ptr in
// its backward_fn). That is the correct and intentional capture so gradient
// computations can read input data. It is not a cycle because those captured
// Tensors are the *inputs*, not the output that owns this Node.
//
// Gradient accumulation protocol (correction to initial plan):
//   backward() must:
//   1. Build topological order of all reachable Nodes (iterative DFS).
//   2. Initialise one "incoming gradient" Tensor per node (the gradient of
//      the loss w.r.t. the output of *this* node that all downstream consumers
//      will accumulate into).
//   3. Traverse nodes in REVERSE topological order (output → leaves). For
//      each node, call backward_fn(accumulated_incoming_grad), which should
//      compute gradients w.r.t. inputs and distribute them by calling
//      accumulate_grad() on each input node's incoming-gradient buffer.
//   This guarantees that by the time a node's backward_fn is called, ALL
//   of its downstream consumers have already contributed to its incoming-
//   gradient buffer — the correct diamond-dependency behaviour.
// ---------------------------------------------------------------------------
struct Node {
    // Nodes of the input tensors of the forward op that produced this node.
    // Owned via shared_ptr: parent nodes must outlive the backward traversal.
    std::vector<std::shared_ptr<Node>> parents;

    // Called during backward() with the accumulated gradient of the loss
    // w.r.t. the output of this node. The closure captures input Tensor
    // data (and their Nodes, through their own shared_ptr) by value.
    //
    // IMPORTANT: implementations of this function MUST run under no_grad()
    // (enforced by backward() itself wrapping every backward_fn call inside
    // a NoGradGuard) so that gradient computations do not accidentally build
    // a second autograd graph on top of the backward pass.
    std::function<void(const Tensor&)> backward_fn;

    // Accumulated incoming gradient from all downstream consumers. Allocated
    // lazily when the first consumer writes into it. backward_fn is called
    // exactly once per node, after all consumers have contributed.
    std::shared_ptr<Tensor> incoming_grad;
};

// ---------------------------------------------------------------------------
// Gradient mode — single global flag (not thread_local; single-threaded
// training loop is assumed for this milestone).
//
// When false, ops build no graph nodes regardless of requires_grad on inputs.
// ---------------------------------------------------------------------------
bool grad_mode_enabled() noexcept;

// ---------------------------------------------------------------------------
// NoGradGuard — RAII wrapper that disables gradient tracking for its scope.
//
// Usage:
//   {
//       auto guard = no_grad();
//       // tensor ops here build no graph
//   }
//   // grad mode restored on guard destruction
// ---------------------------------------------------------------------------
class NoGradGuard {
public:
    NoGradGuard();
    ~NoGradGuard();

    // Non-copyable, non-movable — its lifetime is scoped.
    NoGradGuard(const NoGradGuard&) = delete;
    NoGradGuard& operator=(const NoGradGuard&) = delete;

private:
    bool saved_mode_;
};

// Convenience factory — return by value (NRVO), so writing
//   auto g = no_grad();
// or
//   { auto _ = no_grad(); /* scope */ }
// both work correctly.
NoGradGuard no_grad();

} // namespace rl::tensor
