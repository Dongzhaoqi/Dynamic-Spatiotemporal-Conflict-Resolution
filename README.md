# Dynamic Spatiotemporal Conflict Resolution

This repository contains complete-plan decoding,
finite-capacity evaluation, overload-induced robot grouping, capacity-guided
repair, sequential block coevolution, and state-aligned planning updates.
Precomputed candidate paths and robot observations are inputs.

## Method

Each robot has a selected path, a release offset, and a speed scale. Decoding
produces continuous traversal without intermediate wait decisions. Every path
segment occupies its source cell over a half-open interval with a trailing
protection buffer. Robot occupancy is binary per cell, even when that robot's
protected intervals overlap.

For each complete candidate, the evaluator computes

```text
rho(c,t) = max(number_of_occupying_robots(c,t) - effective_capacity(c,t), 0)
V_cap    = sum_c integral rho(c,t)^2 dt
J_op     = sum_i [lambda_delay * Delta_i + lambda_path * D_i
                  + lambda_speed * (alpha_i - 1)^2] + lambda_time * T_max
```

Candidates are compared lexicographically by `(V_cap, J_op)`. Overloaded
cell-time intervals induce a conflict graph; its connected components are
ordered by aggregate overload participation. The optimizer updates these blocks
sequentially, evaluating every candidate in the complete incumbent plan.
The graph is reconstructed between sweeps. Subsequent planning cycles use the
same algorithm while preserving realized decisions and occupancy history.

## Repository contents

```text
core/
  models.py          Workspace, tasks, decisions, observations and capacities
  decoding.py        Full-plan and remaining-path decoding
  capacity.py        Exact event-based capacity integration and operational cost
  conflict_graph.py  Overload graph, connected components and block ordering
  repair.py          Single-component path, release and speed corrections
  coevolution.py     Sequential block search and Algorithm 1
  online.py          Alignment with observed execution states
docs/
  API.md
tests/               Small algorithm-invariant checks
```

Experiment runners, figure generation, animations, benchmark methods, scene
generators, map assets, datasets and manuscript PDFs are outside this package.
The tests use small abstract cell graphs to check mathematical behavior; they
are not reproductions of the manuscript's experimental scenes.

## Use

Python 3.10 or later is required. The core has no third-party runtime dependency.
Work from the repository root, or optionally install with `python -m pip install -e .`.

```python
from core import OptimizerConfig, Problem, solve

# Build a Problem from your cell graph, tasks and precomputed candidate paths.
# See docs/API.md for the complete data contract.
def plan_tasks(problem: Problem):
    return solve(problem, config=OptimizerConfig(seed=0))
```

For a later update, pass the accepted plan and explicit observed states:

```python
updated = solve(problem, initial=accepted.plan, cycle=observed_cycle)
```

`PlanningResult` contains the accepted `plan`, its full `evaluation`, termination
reason, evaluation count, and within-cycle incumbent `history`. A returned plan
can still have residual overload when its search budget expires; check
`result.evaluation.feasible`.

Run the invariant checks from the repository root:

```bash
python -m unittest discover -s tests -v
```

## Manuscript parameters and implementation scope

Defaults follow Table I and Section IV-A: release offsets in `[0, 60]`, speed
scales in `[0.80, 1.20]`, protection time `5`, population size `30`, up to `10`
sweeps, `150` complete evaluations per block including repair, and `6000` per
cycle. Default time limits are `120 s` initially and `1 s` after an update.
The four operational weights are `1`, `0.01`, `10`, and `0.1`.

The manuscript does not uniquely define the stochastic evolutionary operators.
This implementation explicitly uses discrete crossover/path mutation and bounded
Gaussian continuous mutation, reserving part of each block's budget for repair.
These choices are documented in [PAPER_ALIGNMENT.md](docs/PAPER_ALIGNMENT.md).
The code is a manuscript-aligned reference refactor, not a claim of identical
experimental trajectories, reproduced numerical results, global optimality, or
guaranteed capacity feasibility for every input.

## Attribution

The manuscript title above identifies the accompanying work. The supplied draft
does not establish publication metadata; this repository does not invent author
names, a DOI, or a publication venue. No distribution license has been selected
on the authors' behalf.
