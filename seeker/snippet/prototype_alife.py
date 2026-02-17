#date: 2026-02-17T17:47:51Z
#url: https://api.github.com/gists/5d5bccc4828f46e1064c6719bb6fdd9d
#owner: https://api.github.com/users/TheBarret

"""
BFF - Brainfuck Fission/Fusion
An Artificial Life simulation based on Blaise Agüera y Arcas et al.
"Computational Life: How Well-formed, Self-replicating Programs Emerge from Simple Interaction"
arXiv:2406.19108

Architecture:
- Unified tape: code and data share the same bytearray
- Dual heads: IP (reader/executor) and DP (writer/data pointer)
- 8 opcodes: > < } { + - [ ]
- Symbiosis: organisms interact by tape fusion then fission
- No explicit fitness function: survival of the busiest
"""

import random
import time
import os
from typing import List, Tuple, Optional


# ---------------------------------------------------------------------------
# OPCODE REFERENCE
# ---------------------------------------------------------------------------
# >   move DP right  (writer advances)
# <   move DP left   (writer retreats)
# }   move IP right  (reader skips forward — jump assist / rewind)
# {   move IP left   (reader rewinds)
# +   increment tape[dp]
# -   decrement tape[dp]
# [   if tape[dp] == 0: jump to matching ]
# ]   if tape[dp] != 0: jump back to matching [
# ---------------------------------------------------------------------------

VALID_OPCODES = set(b'><}{+-[]')


# ---------------------------------------------------------------------------
# SECTION 1: THE INTERPRETER
# ---------------------------------------------------------------------------

class BFF_Organism:
    """
    A single BFF organism. Its genome IS its tape.
    Code and data share the same bytearray — self-modification is native.
    """

    def __init__(self, genome: str, max_steps: int = 5000):
        self.tape = bytearray(genome.encode('ascii'))
        self.size = len(self.tape)
        self.max_steps = max_steps

        # Dual heads
        self.ip = 0   # Instruction Pointer — the reader/executor
        self.dp = 0   # Data Pointer       — the writer

        self.steps_executed = 0

        # Pre-build jump map for O(1) bracket lookup
        # NOTE: map is built once. Since +/- can mutate the tape mid-run,
        # this map may go stale. We treat stale jumps as graceful no-ops
        # (the ip just advances past the bracket). This mirrors how the
        # paper handles "broken" organisms — they simply run less efficiently.
        self.jump_map = self._build_jump_map()

    # ------------------------------------------------------------------
    # Jump map
    # ------------------------------------------------------------------

    def _build_jump_map(self) -> dict:
        """
        Pre-compute matching bracket positions.
        Unmatched brackets are silently ignored (organism is 'broken' there).
        """
        stack = []
        jumps = {}
        for i, byte_val in enumerate(self.tape):
            if byte_val == ord('['):
                stack.append(i)
            elif byte_val == ord(']'):
                if stack:
                    start = stack.pop()
                    jumps[start] = i
                    jumps[i] = start
                # Unmatched ] — no entry added; will be treated as no-op
        return jumps

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(self) -> int:
        """
        Run the organism up to max_steps.
        Returns steps executed — this IS the fitness metric.
        Every instruction (including jumps) costs exactly one step.
        """
        self.steps_executed = 0

        while self.steps_executed < self.max_steps:
            # Circular wrap — tape is a ring
            self.ip %= self.size
            self.dp %= self.size

            opcode = self.tape[self.ip]

            # --- Execute opcode ---
            if opcode == ord('>'):
                self.dp += 1                              # Writer moves right

            elif opcode == ord('<'):
                self.dp -= 1                              # Writer moves left

            elif opcode == ord('}'):
                self.ip += 1                              # Reader skips forward
                # Note: IP will be incremented again below → net +2
                # This is intentional: } is a "double-step" for the reader,
                # useful for jumping over sections of code.

            elif opcode == ord('{'):
                self.ip -= 1                              # Reader steps back
                # Net effect: IP stays in place (−1 + 1 at bottom = 0)
                # Use {{{ to rewind. This matches the paper's design intent.

            elif opcode == ord('+'):
                self.tape[self.dp] = (self.tape[self.dp] + 1) % 256

            elif opcode == ord('-'):
                self.tape[self.dp] = (self.tape[self.dp] - 1) % 256

            elif opcode == ord('['):
                if self.tape[self.dp] == 0:
                    # Jump forward to matching ] (or stay if unmatched)
                    self.ip = self.jump_map.get(self.ip, self.ip)

            elif opcode == ord(']'):
                if self.tape[self.dp] != 0:
                    # Jump back to matching [ (or stay if unmatched)
                    self.ip = self.jump_map.get(self.ip, self.ip)

            # Unknown byte (noise) — treated as NOP, IP still advances

            # Always advance IP and count the step
            self.ip += 1
            self.steps_executed += 1

        return self.steps_executed

    # ------------------------------------------------------------------
    # Genome access
    # ------------------------------------------------------------------

    def get_genome(self) -> str:
        """Return current tape as ASCII string (post-execution, may differ from input)."""
        return self.tape.decode('ascii', errors='replace')

    def get_tape_bytes(self) -> bytearray:
        return bytearray(self.tape)  # Return a copy


# ---------------------------------------------------------------------------
# SECTION 2: THE SYMBIOSIS ENGINE
# ---------------------------------------------------------------------------

class SymbiosisEngine:
    """
    Handles organism interaction: fusion → execution → fission.

    The interaction model (faithful to the paper):
      1. FUSION:   Concatenate tape_A + tape_B into one unified tape.
      2. EXECUTE:  Run the fused organism. During this, IP may wander from
                   A's territory into B's, and DP may copy A's code into B's
                   space — or vice versa.
      3. FISSION:  Split the result back at the original boundary.
                   Each half becomes the "child" organism.
      4. EVALUATE: Children that ran more steps than their parents survive.
    """

    def __init__(self, max_steps: int = 5000):
        self.max_steps = max_steps

    def interact(
        self,
        genome_a: str,
        genome_b: str,
    ) -> Tuple[str, str, int]:
        """
        Fuse two organisms, run them, split them apart.

        Returns:
            child_a (str): left half of the post-execution tape
            child_b (str): right half of the post-execution tape
            steps   (int): total steps executed (fitness signal)
        """
        split_point = len(genome_a)

        # --- FUSION ---
        # Sanitize to printable ASCII before fusing — non-ASCII bytes
        # can appear after +/- mutations drift values out of range.
        # We keep the raw bytes but filter at genome boundary crossings.
        fused_genome = genome_a + genome_b

        # Ensure all characters are valid ASCII (0-127); clamp if not
        safe_bytes = bytearray()
        for ch in fused_genome:
            val = ord(ch) if isinstance(ch, str) else ch
            safe_bytes.append(val % 128)  # Wrap to ASCII range
        fused_genome = safe_bytes.decode('ascii', errors='replace')

        fused = BFF_Organism(fused_genome, max_steps=self.max_steps)

        # --- EXECUTION ---
        steps = fused.run()

        # --- FISSION ---
        result_tape = fused.get_tape_bytes()

        # Pad if tape somehow shrank (shouldn't happen but be safe)
        while len(result_tape) < len(fused_genome):
            result_tape.append(0)

        child_a_bytes = result_tape[:split_point]
        child_b_bytes = result_tape[split_point:split_point + len(genome_b)]

        child_a = child_a_bytes.decode('ascii', errors='replace')
        child_b = child_b_bytes.decode('ascii', errors='replace')

        return child_a, child_b, steps


# ---------------------------------------------------------------------------
# SECTION 3: POPULATION MANAGER
# ---------------------------------------------------------------------------

class BFF_Population:
    """
    Manages a soup of BFF organisms.

    Evolution strategy: "Survival of the Busiest"
    - No explicit fitness function beyond steps executed.
    - Organisms that interact and produce busy children persist.
    - Organisms that die quickly get replaced with fresh random noise.
    - Optional background point mutation (set mutation_rate=0 to disable,
      replicating the paper's key result: complexity emerges WITHOUT mutation).
    """

    # Characters used to seed random genomes — the "primordial soup"
    SOUP_CHARS = '><}{+-[]' + ('.' * 8)  # Bias toward noise over structure

    def __init__(
        self,
        pop_size: int = 100,
        genome_length: int = 64,
        max_steps: int = 5000,
        mutation_rate: float = 0.0,   # Paper result: set to 0 to prove mutation unnecessary
        elite_fraction: float = 0.2,
    ):
        self.pop_size = pop_size
        self.genome_length = genome_length
        self.max_steps = max_steps
        self.mutation_rate = mutation_rate
        self.elite_fraction = elite_fraction

        self.engine = SymbiosisEngine(max_steps=max_steps)

        # Seed population with random noise
        self.population: List[str] = [
            self._random_genome() for _ in range(pop_size)
        ]

        # Stats tracking
        self.generation = 0
        self.best_steps_ever = 0
        self.total_steps_this_gen = 0

    # ------------------------------------------------------------------
    # Genome utilities
    # ------------------------------------------------------------------

    def _random_genome(self) -> str:
        """Generate a random genome from the primordial soup alphabet."""
        return ''.join(random.choices(self.SOUP_CHARS, k=self.genome_length))

    def _point_mutate(self, genome: str) -> str:
        """Optional background mutation — single character replacement."""
        if self.mutation_rate == 0.0:
            return genome
        chars = list(genome)
        for i in range(len(chars)):
            if random.random() < self.mutation_rate:
                chars[i] = random.choice(self.SOUP_CHARS)
        return ''.join(chars)

    # ------------------------------------------------------------------
    # One generation
    # ------------------------------------------------------------------

    def evolve_step(self) -> dict:
        """
        Run one generation of the population.

        Process:
          1. Randomly pair organisms for symbiotic interaction.
          2. Run each pair through fusion/execution/fission.
          3. Score children by steps executed.
          4. Keep the busiest, replace the quietest with noise.

        Returns a stats dict for logging/visualization.
        """
        self.generation += 1
        interaction_results = []

        # Shuffle population so pairings are random
        indices = list(range(self.pop_size))
        random.shuffle(indices)

        # Pair them up and interact
        for i in range(0, self.pop_size - 1, 2):
            idx_a = indices[i]
            idx_b = indices[i + 1]

            genome_a = self.population[idx_a]
            genome_b = self.population[idx_b]

            # Apply optional background mutation before interaction
            genome_a = self._point_mutate(genome_a)
            genome_b = self._point_mutate(genome_b)

            child_a, child_b, steps = self.engine.interact(genome_a, genome_b)

            interaction_results.append((child_a, steps, idx_a))
            interaction_results.append((child_b, steps, idx_b))

        # Sort by steps (busiest wins)
        interaction_results.sort(key=lambda x: x[1], reverse=True)

        # Update population
        elite_count = int(self.pop_size * self.elite_fraction)
        new_population = list(self.population)  # Start with current

        for rank, (child, steps, original_idx) in enumerate(interaction_results):
            if rank < elite_count:
                # Elite: keep the child (it's busy)
                new_population[original_idx] = child
            else:
                # Non-elite: replace with noise to maintain diversity
                # (Only replace the bottom fraction)
                bottom_threshold = int(self.pop_size * (1 - self.elite_fraction))
                if rank >= bottom_threshold:
                    new_population[original_idx] = self._random_genome()
                else:
                    new_population[original_idx] = child

        self.population = new_population

        # Stats
        all_steps = [r[1] for r in interaction_results]
        best_steps = max(all_steps)
        avg_steps = sum(all_steps) / len(all_steps)
        self.total_steps_this_gen = sum(all_steps)
        self.best_steps_ever = max(self.best_steps_ever, best_steps)

        best_genome = interaction_results[0][0]

        return {
            'generation':    self.generation,
            'best_steps':    best_steps,
            'avg_steps':     avg_steps,
            'best_ever':     self.best_steps_ever,
            'best_genome':   best_genome,
            'total_steps':   self.total_steps_this_gen,
            'population':    self.population,
            'max_steps':     self.max_steps,
        }


# ---------------------------------------------------------------------------
# SECTION 4: TAPE VISUALIZER
# ---------------------------------------------------------------------------

class TapeVisualizer:
    """
    Renders the population's tapes as a 2D color grid in the terminal.

    Color mapping:
        >  (62)  → Cyan
        <  (60)  → Blue
        }  (125) → Bright Cyan
        {  (123) → Bright Blue
        +  (43)  → Green
        -  (45)  → Red
        [  (91)  → Yellow
        ]  (93)  → Bright Yellow
        other    → Dark grey (noise)
    """

    # ANSI color codes
    ANSI = {
        ord('>'): '\033[36m',    # Cyan
        ord('<'): '\033[34m',    # Blue
        ord('}'): '\033[96m',    # Bright Cyan
        ord('{'): '\033[94m',    # Bright Blue
        ord('+'): '\033[32m',    # Green
        ord('-'): '\033[31m',    # Red
        ord('['): '\033[33m',    # Yellow
        ord(']'): '\033[93m',    # Bright Yellow
    }
    RESET  = '\033[0m'
    NOISE  = '\033[90m'   # Dark grey for non-opcode bytes
    BLOCK  = '█'          # Visual block character for the grid

    def __init__(self, display_width: int = 64):
        self.display_width = display_width

    def render_tape(self, genome: str) -> str:
        """Render a single genome as a colored string."""
        result = []
        for char in genome[:self.display_width]:
            byte_val = ord(char) if char.isprintable() else 0
            color = self.ANSI.get(byte_val, self.NOISE)
            result.append(f"{color}{self.BLOCK}{self.RESET}")
        return ''.join(result)

    def render_population_grid(self, population: List[str], max_rows: int = 20) -> str:
        """Render the top N organisms as a stacked grid."""
        lines = []
        for genome in population[:max_rows]:
            lines.append(self.render_tape(genome))
        return '\n'.join(lines)

    def render_stats(self, stats: dict) -> str:
        """Render generation statistics."""
        gen        = stats['generation']
        best       = stats['best_steps']
        avg        = stats['avg_steps']
        best_ever  = stats['best_ever']
        genome_snip = stats['best_genome'][:32]

        # Phase detection — proportional to max_steps budget
        max_s = stats.get("max_steps", 5000)
        frac = best / max(max_s, 1)
        if frac < 0.02:
            phase = "🌑 Noise — programs dying instantly"
        elif frac < 0.40:
            phase = "🌒 Loop Emergence — structure forming"
        elif frac < 0.95:
            phase = "🌕 Active — sustained computation"
        else:
            phase = "💥 PHASE TRANSITION — Replicator Candidate!"

        return (
            f"\n{'='*66}\n"
            f" Gen: {gen:<6}  Best: {best:<6}  Avg: {avg:<8.1f}  "
            f"Best Ever: {best_ever:<6}\n"
            f" Phase: {phase}\n"
            f" Best Genome: [{genome_snip}...]\n"
            f"{'='*66}"
        )


# ---------------------------------------------------------------------------
# SECTION 5: MAIN SIMULATION LOOP
# ---------------------------------------------------------------------------

def run_simulation(
    pop_size: int       = 100,
    genome_length: int  = 64,
    max_steps: int      = 5000,
    mutation_rate: float= 0.0,   # Keep at 0 to replicate paper's key result
    generations: int    = 500,
    display_rows: int   = 20,
    refresh_every: int  = 1,     # Render every N generations
):
    """
    Main entry point for the BFF simulation.

    Key parameters to experiment with:
        mutation_rate=0.0  → Proves complexity emerges via symbiosis alone
        mutation_rate=0.01 → Adds background noise (faster but less pure)
        max_steps=5000     → Higher = more room for replicators to express
        genome_length=64   → Longer genomes = more complex possible structures
    """
    print("\033[2J\033[H")  # Clear screen
    print("BFF — Brainfuck Fission/Fusion Artificial Life")
    print(f"Pop: {pop_size}  |  Genome: {genome_length}  |  "
          f"Max Steps: {max_steps}  |  Mutation: {mutation_rate}")
    print("Based on: Agüera y Arcas et al., arXiv:2406.19108\n")

    pop = BFF_Population(
        pop_size=pop_size,
        genome_length=genome_length,
        max_steps=max_steps,
        mutation_rate=mutation_rate,
    )

    viz = TapeVisualizer(display_width=64)

    start_time = time.time()
    steps_per_second_history = []

    try:
        for gen in range(generations):
            t0 = time.time()
            stats = pop.evolve_step()
            t1 = time.time()

            # Throughput calculation (MOps/sec)
            elapsed = t1 - t0 if t1 > t0 else 1e-9
            mops = stats['total_steps'] / elapsed / 1_000_000
            steps_per_second_history.append(mops)

            if gen % refresh_every == 0:
                # Move cursor to top (avoid full clear flicker)
                print("\033[H", end='')

                # Render population grid
                grid = viz.render_population_grid(
                    stats['population'],
                    max_rows=display_rows
                )
                print(grid)

                # Render stats
                stat_str = viz.render_stats(stats)
                print(stat_str)
                print(f" Throughput: {mops:.2f} MOps/sec  |  "
                      f"Wall time: {time.time()-start_time:.1f}s")

    except KeyboardInterrupt:
        print("\n\nSimulation interrupted.")

    # Final report
    print(f"\n\nFinal generation: {pop.generation}")
    print(f"Best fitness ever: {pop.best_steps_ever} steps")
    print(f"Peak throughput: {max(steps_per_second_history):.2f} MOps/sec")
    print(f"Average throughput: "
          f"{sum(steps_per_second_history)/len(steps_per_second_history):.2f} MOps/sec")

    return pop


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # --- Experiment A: Pure Symbiogenesis (no mutation) ---
    # This replicates the paper's key claim: complexity emerges from
    # interaction alone. Mutation rate is zero.
    run_simulation(
        pop_size      = 100,
        genome_length = 64,
        max_steps     = 5000,
        mutation_rate = 0.0,    # <-- The paper's key variable
        generations   = 1000,
        display_rows  = 20,
        refresh_every = 1,
    )

    # --- Experiment B: With mutation (uncomment to try) ---
    # run_simulation(
    #     pop_size      = 100,
    #     genome_length = 64,
    #     max_steps     = 5000,
    #     mutation_rate = 0.005,
    #     generations   = 1000,
    # )