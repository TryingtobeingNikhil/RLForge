// Models the scenario in tests/test_version_guard.cpp:27-59:
//
//   auto x = Tensor::from_data({1.0, 2.0, 3.0}, {3});
//   x.requires_grad_(true);
//   auto loss = x.mul(x).mean();
//   auto buf = x.data_mutable();
//   buf[0] = 99.0;
//   loss.backward();   // throws std::runtime_error
//
// and the real error text built by check_version() in src/tensor/tensor.cpp:67-79.

export const VG_X = [1, 2, 3]; // tests/test_version_guard.cpp:30
export const VG_MUTATION = { index: 0, value: 99 }; // tests/test_version_guard.cpp:38

// src/tensor/tensor.cpp:72-77
export function checkVersionMessage(op: string, which: string, saved: number, current: number): string {
  return (
    `${op} backward: ${which} tensor was mutated in-place after the forward pass that created ` +
    `this graph node — stale gradient computation detected. ` +
    `(saved version=${saved}, current version=${current})`
  );
}

// mul's backward checks the rhs operand first (src/tensor/tensor.cpp:524-525).
// For x.mul(x) both operands share one Storage, whose version starts at 0
// (include/rl/tensor/tensor.hpp:43) and is bumped once by the single
// assignment through data_mutable() (include/rl/tensor/tensor.hpp:70-71).
export const VG_THROW_MESSAGE = checkVersionMessage("mul", "rhs", 0, 1);

/** d/dx mean(x * x) = 2x / N, using whatever data the closure sees. */
export function mulMeanGrad(data: number[]): number[] {
  return data.map((v) => (2 * v) / data.length);
}

/** The gradient that matches the forward pass that actually ran. */
export const VG_CORRECT_GRAD = mulMeanGrad(VG_X);

/** What an engine without the guard would silently hand back: the math run on mutated data. */
export const VG_STALE_GRAD = mulMeanGrad(VG_X.map((v, i) => (i === VG_MUTATION.index ? VG_MUTATION.value : v)));
