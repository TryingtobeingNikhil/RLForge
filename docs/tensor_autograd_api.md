# Tensor & Autograd API

This covers Milestone 5. Read earlier docs first — this milestone adds a
self-contained Tensor type with reverse-mode automatic differentiation, and
does NOT require changes to any prior milestone code.

## What this milestone adds

A minimal, correct foundation for neural network layers (Milestone 6) and DQN
(Milestone 7):

- **Tensor** — dense, contiguous, row-major N-D buffer of `double`, with a
  general `std::vector<int64_t>` shape.
- **Core math ops** — `add`, `sub`, `mul` (elementwise and scalar), `matmul`
  (2-D only), `relu`, `square`, `mean`, `gather`.
- **Reverse-mode autograd** — dynamic define-by-run graph, correct diamond-
  dependency handling, `backward()`, `zero_grad()`, `no_grad()`, `detach()`.

## Design choices and rationale

### Tensor storage (`rl::tensor::Tensor`)

```
Tensor
 ├── shared_ptr<Storage>  — contiguous vector<double>, kept alive for backward
 ├── vector<int64_t> shape_
 ├── bool rg_             — whether this tensor participates in autograd
 ├── shared_ptr<Tensor> grad_  — only allocated for leaves with requires_grad
 └── shared_ptr<Node> node_   — NON-NULL for leaves with requires_grad (see below);
                                null for tensors created under no_grad() or without rg
```

`Storage` is wrapped in `shared_ptr` so that backward closures that capture
input tensors by `shared_ptr<Storage>` keep the data alive as long as the
graph exists, even if the user drops their handle to an intermediate result.
This is NOT for view aliasing — every Tensor owns its own contiguous buffer
this milestone.

Scalar type is fixed as `double` throughout. No template parameter.

### Autograd graph (`rl::tensor::Node`)

```
Node
 ├── vector<shared_ptr<Node>> parents    — input nodes (owned, no cycle)
 ├── function<void(const Tensor&)> backward_fn
 └── shared_ptr<Tensor> incoming_grad   — accumulated upstream gradient
```

**Ownership model (no cycles):**

```
output_tensor  --shared_ptr-->  output_node  --shared_ptr-->  parent_nodes
```

Nodes own their parent nodes (the input nodes of the op that produced them).
There is NO back-edge from a Node to any Tensor that owns it — so no reference
cycle exists.

**Leaf nodes — ACTUAL design (deviation from initial spec):**

The initial design spec said `node_ = nullptr` for leaf tensors created with
`requires_grad_(true)`. The implementation **intentionally deviates**: calling
`requires_grad_(true)` on a Tensor creates a **leaf Node** with an empty
`parents` vector and a `backward_fn` that accumulates incoming gradient into
the leaf's `grad_` buffer.

**Why:** backward closures in op Nodes must reference the leaf in order to
deposit gradient. The natural way to do this is to store a `Tensor*` — but
that pointer dangles if the user passes the leaf Tensor by value into a lambda
or it goes out of scope before backward runs. Giving the leaf its own Node,
whose closure captures `grad_` as `shared_ptr<Tensor>`, keeps the gradient
buffer alive for the duration of backward with no dangling pointer.

**Is this a cycle?** No. The ownership graph is:

```
leaf_tensor.node_  ──shared_ptr──►  leaf_Node
                                       │
                   backward_fn captures grad_ptr = leaf_tensor.grad_
                   grad_ptr is a shared_ptr<Tensor> to the GRAD BUFFER,
                   which is a separate Tensor object — NOT leaf_tensor itself.
```

There is no edge from `leaf_Node` back to `leaf_tensor`. Confirmed by
`test_tensor.cpp` Section 10, which uses `std::weak_ptr<Node>` to verify all
graph objects are freed when external handles are dropped.

**Backward algorithm:**

1. Build topological order via iterative post-order DFS (not recursive).
2. Reset all `incoming_grad` buffers on every reachable node.
3. Seed the root node's `incoming_grad` with the initial gradient.
4. Traverse root→leaves. For each node, call `backward_fn(incoming_grad)`
   **only after all downstream consumers have already contributed** to
   `incoming_grad` — this is guaranteed by topological order and is what
   makes diamond dependencies correct (every consumer's gradient is summed
   before the node's own backward function runs).
5. Every `backward_fn` call runs under `no_grad()` to prevent building a
   second autograd graph on top of the backward pass.
6. Gradients accumulate (`+=`) into leaf `grad_` buffers. Call `zero_grad()`
   explicitly — backward does not auto-zero.

### no_grad() guard

```cpp
{
    auto guard = no_grad();  // grad_mode_enabled() is now false
    // ops here build no graph nodes, output.requires_grad() == false
}
// grad mode restored
```

Gradient mode is stored in a `thread_local` bool. Each worker or learner thread
therefore has independent mode state: `no_grad()` in an inference thread cannot
disable graph construction in another thread.

## API reference

### `rl::tensor::Tensor`

| Method | Description |
|--------|-------------|
| `Tensor(shape)` | Zero-filled tensor |
| `Tensor::zeros(shape)` | Factory: zero-filled |
| `Tensor::ones(shape)` | Factory: one-filled |
| `Tensor::from_data(data, shape)` | Factory: from `vector<double>` |
| `shape()`, `ndim()`, `numel()` | Shape accessors |
| `item()` | Extract scalar value (throws if numel≠1) |
| `data()` | Read-only raw buffer access |
| `data_mutable()` | Version-tracked mutable view |
| `operator[](i)` | Bounds-checked flat access; mutable assignments are version-tracked |
| `requires_grad_(bool)` | Enable/disable grad tracking (returns `*this`) |
| `requires_grad()` | Query grad tracking |
| `grad()` | Access leaf gradient buffer (nullptr until backward) |
| `accumulate_grad(g)` | Manually accumulate into leaf grad |
| `zero_grad()` | Reset leaf gradient to zero |
| `backward()` | Run backward (scalar tensors only) |
| `backward(Tensor grad)` | Run backward with explicit upstream grad |
| `detach()` | New tensor sharing data, no graph, requires_grad=false |
| `add(Tensor)`, `+` | Elementwise add |
| `sub(Tensor)`, `-` | Elementwise subtract |
| `mul(Tensor)`, `*` | Elementwise multiply |
| `mul(double)`, `*` | Scalar multiply (both `t * s` and `s * t`) |
| `matmul(Tensor)` | 2-D matrix multiply [M,K]@[K,N]→[M,N] |
| `relu()` | max(0, x); subgradient 0 at x=0 |
| `square()` | x² elementwise |
| `exp()`, `log()` | Elementwise exponential and natural logarithm |
| `clamp(min,max)` | Elementwise clipping with autograd |
| `minimum(Tensor)` | Elementwise minimum with deterministic tie gradients |
| `log_softmax()` | Stable row-wise log-softmax for `[B,C]` |
| `mean()` | Mean over all elements → scalar tensor |
| `gather(indices)` | `out[i] = this[i, indices[i]]` for 2-D input |

### `rl::tensor::no_grad()` / `NoGradGuard`

```cpp
auto g = no_grad();   // all ops until g is destroyed produce no graph
```

## What is explicitly supported

- **0-D, 1-D, 2-D tensors** — all ops work correctly; matmul validates rank 2.
- **Scalar-tensor broadcasting** — `tensor * scalar` and `scalar * tensor`.
- **Multiple backward() calls** — gradients accumulate (+=) across calls into
  the leaf's grad buffer; call `zero_grad()` before each pass to prevent this.
- **Diamond dependencies** — a tensor used as input to two different downstream
  ops receives the correctly summed gradient from both paths.
- **Gather with duplicate indices** — backward correctly scatter-adds into the
  same position rather than overwriting.

## What is explicitly deferred (read before starting Milestone 6)

### General tensor-tensor broadcasting
Needed for `Linear` layer's bias addition (shape `[B, out]` + shape `[out]`).
All tensor-tensor binary ops currently require exact shape matching and throw
`std::invalid_argument` otherwise.

### Views / slicing / strided tensors
Every Tensor owns its own contiguous storage. Memory-efficient batching will
require a strided view mechanism. Not implemented this milestone.

### In-place mutation safety ⚠️

> **Must be addressed before Milestone 6's optimizer work begins.**
>
> Optimizers mutate parameter data in place (`param -= lr * grad`). If `param`
> still has an attached autograd graph from the forward pass, this silently
> produces wrong gradients on the next forward. A version-counter or
> stale-graph guard mechanism must be added before any optimizer implementation.
>
> The code comment in `tensor.hpp` marks this as a known limitation. It is NOT
> safe to use an optimizer that does in-place data mutation until this is fixed.

### Thread safety of no_grad()
Addressed in Milestone 9: gradient mode is thread-local.

### BLAS / SIMD / GPU
Addressed in Milestone 10: `matmul` dispatches to portable CPU, optional CBLAS,
or optional CUDA/cuBLAS backends.

## Extending in Milestone 6

The `Linear` layer will need:
1. **General broadcasting for bias addition** — add a `broadcast_add(bias)`
   method or implement broadcasting in `add` for the `[B, out] + [out]` case
   specifically.
2. **In-place mutation guard** — before the first `SGD` or `Adam` optimizer
   step mutates any `weight.data_mutable()`, the version-counter mechanism
   must exist.
3. `softmax`, `log_softmax`, `cross_entropy` for classification heads.
4. Parameter initialization utilities (`kaiming_uniform_`, `zeros_`, etc.).
