#date: 2025-11-21T16:51:04Z
#url: https://api.github.com/gists/6c454c57b077435da2f7cb989a65156e
#owner: https://api.github.com/users/arbiusone-star

"""
Example miner agent that solves TSP problems.
"""
import asyncio
import sn43
from typing import Dict, Any, List

class TSPMinerAgent(sn43.Agent):
    """Base agent class for TSP miners."""
    
    def __init__(self):
        super().__init__()
        print(f"TSP Miner Agent initialized", flush=True)
        self.solve_count = 0
    
    def init(self, ctx: sn43.Context) -> None:
        print(f"TSP Miner Agent init called with context", flush=True)
    
    @sn43.entrypoint
    async def solve_problem(
        self, 
        problem_data: Dict[str, Any],
        timeout: float = 60.0
    ) -> Dict[str, Any]:
        """
        Solve a TSP problem.
        
        Args:
            ctx: Execution context
            problem_data: Problem specification from generate_problem
        
        Returns:
            Dictionary with solution and metadata
        """
        import sys
        print(f"[MINER] solve_problem called with {problem_data.get('n_nodes')} nodes", file=sys.stderr, flush=True)

        # Example: Call greedy solver
        solution = await self._greedy_solve(problem_data)

        print(f"[MINER] Returning solution", file=sys.stderr, flush=True)
        
        return {
            "solution": solution,
        }
    
    async def _greedy_solve(self, problem_data: Dict[str, Any]) -> List[Any]:
        """Implement solving logic."""
        import sys
        print(f"[MINER] _greedy_solve called", file=sys.stderr, flush=True)

        try:
            # LAZY IMPORT - only import when actually solving
            from sn43.graphite.utils.constants import BENCHMARK_SOLUTIONS, PROBLEM_TYPE
            print(f"[MINER] Imports successful", file=sys.stderr, flush=True)

            n_nodes = problem_data["n_nodes"]
            problem_type = problem_data["problem_type"]
            print(f"[MINER] Problem type: {problem_type}, nodes: {n_nodes}", file=sys.stderr, flush=True)

            problem_formulation = PROBLEM_TYPE.get(problem_type)
            greedy_solver_class = BENCHMARK_SOLUTIONS.get(problem_type)

            if greedy_solver_class:
                print(f"[MINER] Creating problem instance", file=sys.stderr, flush=True)
                problem = problem_formulation(**problem_data)

                print(f"[MINER] Creating solver", file=sys.stderr, flush=True)
                greedy_solver = greedy_solver_class([problem])

                print(f"[MINER] Recreating edges", file=sys.stderr, flush=True)
#                 problem.edges = await sn43.tools.recreate_edges(problem)

                print(f"[MINER] Starting solver.solve_problem", file=sys.stderr, flush=True)
                solution = await asyncio.wait_for(
                    greedy_solver.solve_problem(problem),
                    timeout=30
                )
                print(f"[MINER] Solver completed", file=sys.stderr, flush=True)

                return solution
            else:
                print(f"[MINER] Using default solution", file=sys.stderr, flush=True)
                return list(range(n_nodes))

        except Exception as e:
            print(f"[MINER ERROR] {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            import traceback
            traceback.print_exc(file=sys.stderr)
            raise
    
# Instantiate the agent
print(f"About to instantiate agent", flush=True)
agent = TSPMinerAgent()
print(f"Agent instantiated successfully", flush=True)