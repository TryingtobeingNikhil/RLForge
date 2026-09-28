// A TypeScript port of std::mt19937 plus the two distributions RLForge's
// tabular agent and GridWorld draw from.
//
// The engine is bit-exact with the C++ standard (tests/mt19937.test.ts checks
// the standard's 10000th-output value). The distributions follow libc++ (the
// standard library behind Apple clang). With them, the GridWorld port
// reproduces a native run of the C++ pipeline step for step; see
// scripts/parity/README.md. libstdc++ (GCC) implements these distributions
// differently, so a Linux build of the C++ test takes a different (equally
// valid) trajectory to the same -100 -> 3.0 result.

const N = 624;
const M = 397;

export class MT19937 {
  private readonly mt = new Uint32Array(N);
  private index = N + 1;

  constructor(seed = 5489) {
    this.seed(seed);
  }

  seed(seed: number): void {
    this.mt[0] = seed >>> 0;
    for (let i = 1; i < N; i++) {
      const prev = this.mt[i - 1] ^ (this.mt[i - 1] >>> 30);
      this.mt[i] = (Math.imul(1812433253, prev) + i) >>> 0;
    }
    this.index = N;
  }

  nextU32(): number {
    if (this.index >= N) this.twist();
    let y = this.mt[this.index++];
    y ^= y >>> 11;
    y ^= (y << 7) & 0x9d2c5680;
    y ^= (y << 15) & 0xefc60000;
    y ^= y >>> 18;
    return y >>> 0;
  }

  private twist(): void {
    const mt = this.mt;
    for (let i = 0; i < N; i++) {
      const y = (mt[i] & 0x80000000) | (mt[(i + 1) % N] & 0x7fffffff);
      let v = mt[(i + M) % N] ^ (y >>> 1);
      if (y & 1) v ^= 0x9908b0df;
      mt[i] = v >>> 0;
    }
    this.index = 0;
  }
}

const TWO_POW_32 = 4294967296;

// std::uniform_real_distribution<float>(0, 1) as libc++ computes it:
// generate_canonical<float, 24> takes one 32-bit draw, converts it to float,
// then divides by 2^32 in float.
export function uniformFloat01(rng: MT19937): number {
  return Math.fround(Math.fround(rng.nextU32()) / TWO_POW_32);
}

// std::uniform_int_distribution<int64_t>(0, n - 1) as libc++ computes it for
// a power-of-two range over a 32-bit engine: its independent-bits engine
// keeps the low log2(n) bits of a single draw, with no rejection. RLForge
// only ever draws from Discrete(4), so that is the only case supported.
export function uniformIntBelow(rng: MT19937, n: number): number {
  if (!Number.isInteger(n) || n < 1 || n > 2 ** 31 || (n & (n - 1)) !== 0) {
    throw new RangeError(`uniformIntBelow supports power-of-two ranges only, got ${n}`);
  }
  return (rng.nextU32() & (n - 1)) >>> 0;
}
