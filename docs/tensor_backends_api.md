# Tensor Compute Backends (Milestone 10)

RLForge matrix multiplication now dispatches through a `TensorBackend`.
Tensor storage, public APIs, graph construction, version guards, and numerical
semantics remain unchanged. Forward and backward matrix multiplications use
the backend snapshot selected during the forward operation.

## Backends

- `cpu`: always built and selected by default; portable row-major C++ kernel.
- `blas`: optional CBLAS `dgemm` implementation.
- `cuda`: optional CUDA/cuBLAS implementation. Existing host-backed tensors
  are staged to device memory for the kernel and copied back afterward.

The CUDA implementation prioritizes a compatible, explicit backend boundary.
Persistent device-resident tensor storage and fused kernels are future
optimizations that can now be introduced behind this interface.

## Selecting a backend

```cpp
rl::tensor::set_backend(rl::tensor::make_cpu_backend());

if (rl::tensor::blas_backend_available()) {
    rl::tensor::set_backend(rl::tensor::make_blas_backend());
}
```

`set_backend` rejects null pointers. Selection is synchronized, and each
operation retains a shared pointer for its full lifetime.

## Configuration

Portable CPU only:

```bash
cmake -S . -B build
```

Enable CBLAS:

```bash
cmake -S . -B build -DRL_ENABLE_BLAS=ON
```

On macOS this option links the system Accelerate framework. CUDA is optional
and is not required for a macOS build; the portable CPU backend remains the
default on every platform.

Enable CUDA and cuBLAS:

```bash
cmake -S . -B build -DRL_ENABLE_CUDA=ON
```

Optional backend factories throw a clear runtime error when their backend was
not compiled in. A CUDA build also reports unavailable at runtime when no CUDA
device is present. CUDA configuration requires CMake's `CUDAToolkit` package;
BLAS configuration uses Accelerate on macOS and requires CBLAS elsewhere.
