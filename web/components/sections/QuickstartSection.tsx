import { CodeBlock } from "../CodeBlock";
import { Section, SectionHeader, Src } from "../Section";
import { DOCS, REPO } from "@/lib/content";
import { ArrowIcon } from "../Icons";

// Verbatim from README.md:100-116 ("Quickstart")
const QUICKSTART = `#include "rl/envs/grid_world.hpp"
#include "rl/vector_envs/sync_vector_environment.hpp"
#include "rl/agents/tabular_q_learning_agent.hpp"
#include "rl/core/trainer.hpp"

using namespace rl;

std::vector<vector_envs::EnvFactory> factories = {
    [] { return std::make_unique<envs::GridWorld>(); }
};
vector_envs::SyncVectorEnvironment train_env(factories);
envs::GridWorld eval_env;

agents::TabularQLearningAgent agent(train_env.action_space());
core::Trainer trainer(train_env, eval_env, agent);

auto result = trainer.train(20000);`;

// Condensed from tests/test_trainer_grid_world.cpp:52-89: the statements are
// verbatim; explanatory comments and REQUIREs are trimmed, and the trailing
// comments give the values the test asserts at :76 and :96.
const EXACT_CONFIG = `GridWorld::Config config;
config.size = 5;
config.max_episode_steps = 100;
config.slip_probability = 0.0f;

SyncVectorEnvironment train_env(make_grid_world_factories(4, config));
GridWorld eval_env(config);

TabularQLearningConfig agent_config;
agent_config.learning_rate = 0.1f;
agent_config.discount_factor = 0.99f;
agent_config.epsilon_start = 1.0f;
agent_config.epsilon_end = 0.05f;
agent_config.epsilon_decay_steps = 20000;
agent_config.seed = 0;
TabularQLearningAgent agent(train_env.action_space(), agent_config);

Trainer trainer(train_env, eval_env, agent);
auto baseline = trainer.evaluate(/*num_episodes=*/10, /*seed=*/1000);  // -100.0
auto train_result = trainer.train(/*num_steps=*/20000, /*seed=*/42);
auto after_training = trainer.evaluate(/*num_episodes=*/10, /*seed=*/1000);  // 3.0`;

// Verbatim from README.md:178-182 ("Building it")
const BUILD = `git clone https://github.com/TryingtobeingNikhil/RLForge.git
cd RLForge

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)`;

const TEST = `cd build && ctest --output-on-failure`;

// docs/tensor_backends_api.md "Configuration"
const BACKENDS = `# Enable CBLAS (links Accelerate on macOS)
cmake -S . -B build -DRL_ENABLE_BLAS=ON

# Enable CUDA and cuBLAS
cmake -S . -B build -DRL_ENABLE_CUDA=ON`;

export function QuickstartSection() {
  return (
    <Section id="quickstart">
      <SectionHeader id="quickstart" eyebrow="Quickstart" title="The real API. No hidden setup." aside="No config files. Just the headers and a loop.">
        {/* README.md:96-97 */}
        <p>This is what training an agent actually looks like, straight from the README.</p>
      </SectionHeader>

      <div className="mt-12 grid grid-cols-1 gap-6 lg:grid-cols-[minmax(0,1.15fr)_minmax(0,1fr)]">
        <div className="min-w-0 space-y-6" data-reveal>
          <CodeBlock code={QUICKSTART} title="train.cpp" source={<Src href={`${REPO.blob}/README.md#quickstart`}>README.md, Quickstart</Src>} />
          <details className="group card p-4 sm:p-5">
            <summary className="flex cursor-pointer list-none items-center justify-between gap-3 text-[0.95rem] font-semibold text-ink">
              The exact config behind −100 → 3.0
              <span className="font-mono text-xs text-ink-3 transition-transform duration-300 group-open:rotate-90">▸</span>
            </summary>
            <p className="mt-3 text-sm leading-relaxed text-ink-2">
              The quickstart runs on defaults: one env, GridWorld&apos;s 0.1 slip chance, ε decaying over 10,000 steps, no seeds. The numbers
              on this page come from the test&apos;s pinned config: 4 lanes, no slip, every seed fixed.
            </p>
            <div className="mt-4">
              <CodeBlock
                code={EXACT_CONFIG}
                source={<Src href={`${REPO.blob}/tests/test_trainer_grid_world.cpp#L51-L102`}>tests/test_trainer_grid_world.cpp:52-96, condensed</Src>}
              />
            </div>
          </details>
        </div>

        <div className="min-w-0 space-y-6" data-reveal style={{ ["--reveal-delay" as string]: "120ms" }}>
          <CodeBlock code={BUILD} lang="bash" title="build" />
          <CodeBlock code={TEST} lang="bash" title="run the tests" />
          <div className="card p-4 sm:p-5">
            <p className="text-sm font-semibold text-ink">You&apos;ll need</p>
            {/* README.md:193-197 */}
            <ul className="mt-2 space-y-1 text-sm text-ink-2">
              <li>A C++20 compiler (Clang 14+ / GCC 12+)</li>
              <li>CMake 3.20+</li>
              <li>An internet connection the first time, since Catch2 is fetched via FetchContent</li>
            </ul>
          </div>
          <CodeBlock
            code={BACKENDS}
            lang="bash"
            title="optional backends (CPU stays the default)"
            source={<Src href={`${REPO.blob}/docs/tensor_backends_api.md`}>docs/tensor_backends_api.md</Src>}
          />
        </div>
      </div>

      <div className="mt-16" data-reveal>
        <h3 className="text-2xl font-semibold tracking-[-0.01em] text-ink">Documentation</h3>
        <p className="mt-2 text-[0.95rem] text-ink-2">One API doc per subsystem, on GitHub.</p>
        <ul className="mt-6 grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {DOCS.map((d) => (
            <li key={d.file}>
              <a href={`${REPO.blob}/docs/${d.file}`} className="card spot group flex h-full items-start justify-between gap-3 p-4 transition-colors hover:border-line-2">
                <span className="min-w-0">
                  <span className="block truncate font-mono text-[0.82rem] text-ink">{d.file}</span>
                  <span className="mt-1 block text-sm text-ink-3">{d.covers}</span>
                </span>
                <ArrowIcon className="mt-0.5 h-4 w-4 shrink-0 text-ink-3 transition-transform duration-300 ease-spring group-hover:translate-x-1 group-hover:text-amber" />
              </a>
            </li>
          ))}
        </ul>
      </div>
    </Section>
  );
}
