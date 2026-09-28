// A deliberately tiny reverse-mode autograd, just big enough to run the
// diamond graph from the README and tests/test_tensor.cpp:605-637:
//
//         x              a = x * 2,  b = x * 3
//        / \             c = a + b
//       a   b            loss = mean(c)
//        \ /
//         c
//         |
//       loss
//
// It can deposit gradients two ways: "accumulate" (what RLForge's
// distribute_grad does, src/tensor/tensor.cpp:119-133) or "overwrite" (the
// bug the README describes). Every number the widget shows comes from here.

export type Mode = "accumulate" | "overwrite";
export type NodeId = "x" | "a" | "b" | "c" | "loss";

// tests/test_tensor.cpp:622 — auto x = Tensor::from_data({1.0, 2.0, 3.0, 4.0}, {4});
export const DIAMOND_X = [1, 2, 3, 4];

type Vec = number[];

interface Edge {
  from: NodeId; // the consumer, whose gradient flows back...
  to: NodeId; // ...into this input
  local: (upstream: Vec) => Vec; // vector-Jacobian product for this edge
}

export interface ForwardValues {
  x: Vec;
  a: Vec;
  b: Vec;
  c: Vec;
  loss: number;
}

export interface Deposit {
  edge: `${NodeId}->${NodeId}`;
  from: NodeId;
  to: NodeId;
  contribution: Vec;
  before: Vec | null;
  after: Vec;
}

export interface BackwardTrace {
  mode: Mode;
  deposits: Deposit[];
  grads: Record<NodeId, Vec | null>;
}

const mean = (v: Vec) => v.reduce((s, e) => s + e, 0) / v.length;

export function forward(x: Vec = DIAMOND_X): ForwardValues {
  const a = x.map((e) => e * 2);
  const b = x.map((e) => e * 3);
  const c = a.map((e, i) => e + b[i]);
  return { x, a, b, c, loss: mean(c) };
}

function edges(n: number): Edge[] {
  return [
    { from: "loss", to: "c", local: (g) => new Array(n).fill(g[0] / n) }, // d mean / d c_i = 1/N
    { from: "c", to: "a", local: (g) => g.slice() }, // d(a+b)/da = 1
    { from: "c", to: "b", local: (g) => g.slice() }, // d(a+b)/db = 1
    { from: "a", to: "x", local: (g) => g.map((e) => e * 2) }, // d(2x)/dx = 2
    { from: "b", to: "x", local: (g) => g.map((e) => e * 3) }, // d(3x)/dx = 3
  ];
}

// Nodes are processed in topological order, root first. Within the diamond,
// a's branch is processed before b's, matching the README's telling: a's
// deposit reaches x first and b's "a moment later".
const ORDER: NodeId[] = ["loss", "c", "a", "b", "x"];

export function backward(mode: Mode, x: Vec = DIAMOND_X): BackwardTrace {
  const grads: Record<NodeId, Vec | null> = { x: null, a: null, b: null, c: null, loss: [1] };
  const deposits: Deposit[] = [];
  const all = edges(x.length);
  for (const node of ORDER) {
    const upstream = grads[node];
    if (!upstream) continue;
    for (const e of all.filter((edge) => edge.from === node)) {
      const contribution = e.local(upstream);
      const before = grads[e.to];
      const after =
        mode === "accumulate" && before ? before.map((v, i) => v + contribution[i]) : contribution.slice();
      grads[e.to] = after;
      deposits.push({ edge: `${e.from}->${e.to}`, from: e.from, to: e.to, contribution, before, after });
    }
  }
  return { mode, deposits, grads };
}

// tests/test_tensor.cpp:33 — constexpr double kNumGradEps = 1e-5;
export const NUMGRAD_EPS = 1e-5;

/** Central difference, exactly as numerical_gradient() in tests/test_tensor.cpp:38-54. */
export function numericalGradient(x: Vec = DIAMOND_X, eps: number = NUMGRAD_EPS): Vec {
  return x.map((_, i) => {
    const plus = x.slice();
    const minus = x.slice();
    plus[i] += eps;
    minus[i] -= eps;
    return (forward(plus).loss - forward(minus).loss) / (2 * eps);
  });
}

/** Relative error, as check_grad() computes it (tests/test_tensor.cpp:75-77). */
export function relativeError(analytic: number, numeric: number): number {
  const denom = Math.max(Math.abs(analytic), Math.abs(numeric));
  return denom < 1e-8 ? Math.abs(analytic - numeric) : Math.abs(analytic - numeric) / denom;
}
