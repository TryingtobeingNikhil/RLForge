// Every fact on the site lives here or next to where it's rendered, with a
// comment naming the file it comes from. Paths are relative to the repo root.

export const REPO = {
  owner: "TryingtobeingNikhil",
  name: "RLForge",
  url: "https://github.com/TryingtobeingNikhil/RLForge",
  blob: "https://github.com/TryingtobeingNikhil/RLForge/blob/main",
  tree: "https://github.com/TryingtobeingNikhil/RLForge/tree/main",
  commit: "https://github.com/TryingtobeingNikhil/RLForge/commit",
} as const;

export const PEOPLE = {
  nikhil: { name: "Nikhil Mourya", handle: "TryingtobeingNikhil", url: "https://github.com/TryingtobeingNikhil" },
  aprv10: { name: "aprv10", handle: "aprv10", url: "https://github.com/aprv10" },
} as const;

// README.md:7 (### heading under the title)
export const TAGLINE = "We built the tensor engine so we didn't have to trust anyone else's gradients.";

// ---------------------------------------------------------------------------
// Test suite. Counted from tests/*.cpp (one TEST_CASE per test; the contract
// helpers in tests/contract/ define no TEST_CASEs). tests/test_count.test.ts
// re-counts the C++ sources and fails if any number here drifts.
// ---------------------------------------------------------------------------
export interface TestFile {
  file: string;
  count: number;
}

export interface TestGroup {
  id: string;
  label: string;
  blurb: string;
  files: TestFile[];
}

export const TEST_GROUPS: TestGroup[] = [
  {
    id: "tensor",
    label: "Tensor & autograd",
    blurb: "Every op: forward value, analytic backward, numerical check. Plus diamonds and memory.",
    files: [{ file: "tests/test_tensor.cpp", count: 49 }],
  },
  {
    id: "dqn",
    label: "DQN",
    blurb: "QNetwork, bootstrap masks, ε-greedy, target sync, max_last_dim, bandit convergence.",
    files: [
      { file: "tests/test_dqn.cpp", count: 15 },
      { file: "tests/test_target_network.cpp", count: 5 },
      { file: "tests/test_max_last_dim.cpp", count: 5 },
    ],
  },
  {
    id: "nn",
    label: "NN layers, losses & optimizers",
    blurb: "Broadcasting, transpose, Linear, mse_loss, SGD/Adam by hand, end-to-end fits.",
    files: [
      { file: "tests/test_linear.cpp", count: 12 },
      { file: "tests/test_optim.cpp", count: 5 },
      { file: "tests/test_e2e_linear.cpp", count: 2 },
    ],
  },
  {
    id: "replay",
    label: "Transitions & replay buffer",
    blurb: "Ring buffer, seeded sampling, final-observation handling, a second storage backend.",
    files: [
      { file: "tests/test_replay_buffer.cpp", count: 8 },
      { file: "tests/test_transition.cpp", count: 3 },
    ],
  },
  {
    id: "env",
    label: "Environments & spaces",
    blurb: "Discrete/Box validation, GridWorld termination vs truncation, the shared contract.",
    files: [
      { file: "tests/test_space.cpp", count: 6 },
      { file: "tests/test_grid_world.cpp", count: 4 },
    ],
  },
  {
    id: "vector",
    label: "Vector environments",
    blurb: "Sync and threaded: contract, auto-reset, terminal obs, failures, per-thread grad mode.",
    files: [
      { file: "tests/test_sync_vector_environment.cpp", count: 4 },
      { file: "tests/test_threaded_vector_environment.cpp", count: 4 },
    ],
  },
  {
    id: "tabular",
    label: "Tabular Q-learning & Trainer",
    blurb: "Tie-breaking, bootstrap rules, ε decay, and the −100 → 3.0 GridWorld run.",
    files: [
      { file: "tests/test_tabular_q_learning_agent.cpp", count: 6 },
      { file: "tests/test_trainer_grid_world.cpp", count: 2 },
    ],
  },
  {
    id: "ppo",
    label: "PPO",
    blurb: "GAE termination vs truncation, lane independence, rollouts, log_softmax gradients.",
    files: [
      { file: "tests/test_ppo.cpp", count: 4 },
      { file: "tests/test_ppo_tensor_ops.cpp", count: 3 },
    ],
  },
  {
    id: "guard",
    label: "Version guard",
    blurb: "In-place mutation after forward throws; no false positives; no bypass via views.",
    files: [{ file: "tests/test_version_guard.cpp", count: 6 }],
  },
  {
    id: "backend",
    label: "Compute backends",
    blurb: "Dispatch, backward snapshot, CPU always on, optional backends fail loudly.",
    files: [{ file: "tests/test_tensor_backend.cpp", count: 5 }],
  },
];

export const groupTotal = (g: TestGroup) => g.files.reduce((s, f) => s + f.count, 0);
export const TOTAL_TESTS = TEST_GROUPS.reduce((s, g) => s + groupTotal(g), 0); // 148, README.md:12 badge

// Catch2 tags, counted over all 148 TEST_CASE declarations. A test can carry
// several tags, so these overlap and don't sum to 148.
export const TEST_TAGS: { tag: string; count: number }[] = [
  { tag: "tensor", count: 59 },
  { tag: "autograd", count: 22 },
  { tag: "ops", count: 15 },
  { tag: "numgrad", count: 13 },
  { tag: "version_guard", count: 8 },
  { tag: "replay_buffer", count: 8 },
  { tag: "ppo", count: 7 },
  { tag: "space", count: 6 },
  { tag: "tabular_q_learning", count: 6 },
  { tag: "qnetwork", count: 5 },
  { tag: "max_last_dim", count: 5 },
  { tag: "optim", count: 5 },
  { tag: "target_network", count: 5 },
  { tag: "backend", count: 5 },
  { tag: "dqn", count: 4 },
  { tag: "integration", count: 4 },
  { tag: "grid_world", count: 4 },
  { tag: "broadcast", count: 4 },
  { tag: "vector_env", count: 4 },
  { tag: "memory", count: 4 },
  { tag: "threaded_vector_env", count: 4 },
  { tag: "batch_to_tensors", count: 3 },
  { tag: "epsilon_greedy", count: 3 },
  { tag: "e2e", count: 3 },
  { tag: "contract", count: 3 },
  { tag: "linear", count: 3 },
  { tag: "transition", count: 3 },
  { tag: "loss", count: 2 },
  { tag: "sgd", count: 2 },
  { tag: "gae", count: 2 },
  { tag: "trainer", count: 2 },
  { tag: "adam", count: 1 },
];

// ---------------------------------------------------------------------------
// Build log. Titles and "the actual hard part" are verbatim from the README's
// build-log table (README.md:140-149); commits from `git log`.
// ---------------------------------------------------------------------------
export type Builder = "nikhil" | "both";

export interface Milestone {
  n: number;
  title: string;
  hardPart: string;
  shipped: string[];
  doc: string | null;
  commit: { sha: string; label: string };
  builtBy: Builder;
}

export const MILESTONES: Milestone[] = [
  {
    n: 1,
    title: "Environment Interface",
    hardPart: "Getting `terminated`/`truncated` right before anything else depended on getting it wrong",
    shipped: ["Environment + EnvironmentBase", "Discrete / Box spaces", "GridWorld reference env"],
    doc: "docs/environment_api.md",
    commit: { sha: "88859df", label: "Day 1: Initial setup and C++ project structure" },
    builtBy: "nikhil",
  },
  {
    n: 2,
    title: "Vector Environment",
    hardPart: "Auto-reset that preserves the real final observation instead of quietly discarding it",
    shipped: ["VectorEnvironment", "SyncVectorEnvironment", "shared contract tests"],
    doc: "docs/vector_environment_api.md",
    commit: { sha: "a249992", label: "Day 2: Added VectorEnvironment and SyncVectorEnvironment" },
    builtBy: "nikhil",
  },
  {
    n: 3,
    title: "Replay Buffer & Transitions",
    hardPart: "A storage interface the sampling logic doesn't need to know or care about",
    shipped: ["Transition / make_transitions", "ReplayBuffer", "TransitionStorage strategy"],
    doc: "docs/replay_buffer_api.md",
    commit: { sha: "b27464a", label: "Day 3: Added ReplayBuffer and Transition API" },
    builtBy: "nikhil",
  },
  {
    n: 4,
    title: "Agent & Trainer",
    hardPart:
      "One training loop that runs step-based Q-Learning and rollout-based PPO without either one bending to fit the other",
    shipped: ["Agent's four hooks", "Trainer train/evaluate", "TabularQLearningAgent"],
    doc: "docs/agent_trainer_api.md",
    commit: { sha: "d742cd9", label: "Day 4: Consolidate Milestone 4" },
    builtBy: "nikhil",
  },
  {
    n: 5,
    title: "Tensor & Autograd",
    hardPart: "The diamond-dependency problem above, solved and numerically proven, not assumed",
    shipped: ["Tensor on shared Storage", "reverse-mode autograd", "no_grad / detach"],
    doc: "docs/tensor_autograd_api.md",
    commit: { sha: "aa32f3d", label: "Milestone 5: Tensor + Core Math + Reverse-Mode Autograd" },
    builtBy: "nikhil",
  },
  {
    n: 6,
    title: "NN Layers & Optimizers",
    hardPart: "Kaiming init, SGD with momentum, Adam, and broadcasting, all running on our own tensor engine",
    shipped: ["Module / Linear", "SGD + Adam", "version-counter guard"],
    doc: "docs/nn_optim_api.md",
    commit: { sha: "d9c4727", label: "Milestone 6: Neural Network Layers & Optimisers" },
    builtBy: "nikhil",
  },
  {
    n: 7,
    title: "DQN",
    hardPart: "Bootstrap masking that respects terminated vs. truncated all the way through the target computation",
    shipped: ["DQNAgent", "QNetwork + target net", "batch_to_tensors"],
    doc: "docs/dqn_api.md",
    commit: { sha: "1abd060", label: "Milestone 7: Deep Q-Network (DQN)" },
    builtBy: "nikhil",
  },
  {
    n: 8,
    title: "PPO",
    hardPart: "Generalized advantage estimation with lane-correct handling across a batch of parallel environments",
    shipped: ["PPOAgent", "ActorCriticNetwork", "RolloutBuffer + GAE"],
    doc: "docs/ppo_api.md",
    commit: { sha: "fab9545", label: "feat: implement PPO and parallel rollout backends" },
    builtBy: "both",
  },
  {
    n: 9,
    title: "Multi-threaded Rollouts",
    hardPart:
      "Persistent worker threads, deterministic ordering, and gradient-mode isolation that doesn't leak across threads",
    shipped: ["ThreadedVectorEnvironment", "thread-local grad mode", "exception propagation"],
    doc: "docs/threaded_vector_environment_api.md",
    commit: { sha: "fab9545", label: "feat: implement PPO and parallel rollout backends" },
    builtBy: "both",
  },
  {
    n: 10,
    title: "CUDA / CBLAS Backends",
    hardPart: "Swapping the matmul kernel underneath the whole autograd graph without the graph noticing",
    shipped: ["TensorBackend dispatch", "CBLAS dgemm", "CUDA / cuBLAS"],
    doc: "docs/tensor_backends_api.md",
    commit: { sha: "fab9545", label: "feat: implement PPO and parallel rollout backends" },
    builtBy: "both",
  },
];

// README.md "Documentation" table, plus docs/vector_environment_api.md, which
// exists in docs/ but is missing from that table.
export const DOCS: { file: string; covers: string }[] = [
  { file: "environment_api.md", covers: "Environment, Space, VectorEnvironment" },
  { file: "vector_environment_api.md", covers: "Batched reset/step, auto-reset, final observations" },
  { file: "replay_buffer_api.md", covers: "Transition, ReplayBuffer" },
  { file: "agent_trainer_api.md", covers: "Agent, Trainer, TabularQLearning" },
  { file: "tensor_autograd_api.md", covers: "Tensor, autograd, and its documented limitations" },
  { file: "nn_optim_api.md", covers: "Neural layers and optimizers" },
  { file: "dqn_api.md", covers: "Deep Q-Network" },
  { file: "ppo_api.md", covers: "PPO and generalized advantage estimation" },
  { file: "threaded_vector_environment_api.md", covers: "Persistent threaded rollouts" },
  { file: "tensor_backends_api.md", covers: "CPU, CBLAS, and CUDA backends" },
];

// "What we didn't build (yet)". The first three are the README's own list
// (README.md "What we didn't build (yet)"); the rest are deferrals the docs
// state in so many words.
export const NOT_BUILT: { item: string; source: string }[] = [
  { item: "Strided / view tensors", source: "README.md" },
  { item: "General N-D broadcasting beyond what DQN and PPO need today", source: "README.md" },
  { item: "Continuous-action policies: everything here is discrete-action for now", source: "README.md" },
  { item: "Recurrent policies, observation normalization, checkpoint serialization", source: "docs/ppo_api.md" },
  { item: "Persistent device-resident tensor storage and fused CUDA kernels", source: "docs/tensor_backends_api.md" },
  { item: "A logging framework: results come back as raw series and metrics", source: "docs/agent_trainer_api.md" },
];
