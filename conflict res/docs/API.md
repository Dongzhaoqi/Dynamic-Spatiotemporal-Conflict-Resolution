# Input and API guide

The core accepts a generic finite cell graph representing a three-dimensional
workspace. It does not generate a scene, discretize geometry, compute low-level
controls, or search for candidate paths. Supply those application inputs before
calling the planner.

## Construct the problem

```python
from core import Bounds, ObjectiveWeights, Problem, RobotTask, Workspace

workspace = Workspace(
    capacities=cell_capacities,       # dict[str, positive int]
    distances=transition_lengths,    # dict[tuple[str, str], positive float]
)
tasks = tuple(
    RobotTask(
        robot_id=item["robot_id"],   # positive integer; defines tie-break order
        paths=item["candidate_paths"],  # sequences of cell IDs
        nominal_speed=item["nominal_speed"],
        scheduled_release=item["scheduled_release"],
    )
    for item in task_data
)
problem = Problem(
    workspace=workspace,
    tasks=tasks,
    horizon=planning_horizon,
    protection_time=5.0,
    bounds=Bounds(max_delay=60.0, speed_min=0.80, speed_max=1.20),
    weights=ObjectiveWeights(delay=1.0, path_length=0.01, speed=10.0, makespan=0.1),
)
```

The variables in this integration snippet are caller-supplied data, not bundled
experimental inputs. Use consistent distance, time and speed units. The default
parameters assume the manuscript's seconds and compatible distance/speed units.
An edge is directed: supply both `(a, b)` and `(b, a)` when both directions are
admissible. Paths must follow supplied positive-length edges and share the task's
origin and destination. The paper's experiments used at most three candidate
paths; the mathematical model and API allow any finite nonempty set.

The finite `horizon` must contain every protected completion reachable under
the current admissible domain, not only the initial plan. The decoder rejects
short horizons instead of clipping away violations. `required_horizon(problem,
cycle)` returns the conservative bound used by the decoder. A `Problem` can be
constructed first and its horizon adjusted with `dataclasses.replace` if needed.

For unreleased robots, the bound is the largest scheduled release plus maximum
delay plus the longest candidate duration at minimum speed scale, plus the
protection time. For active robots it uses the update time and the longest
admissible remaining path. Supplied historical protection tails must also fit.

## Plan and evaluation

```python
from core import Decision, OptimizerConfig, decode, evaluate, solve

initial = {task.robot_id: Decision() for task in problem.tasks}
baseline = evaluate(problem, initial)
result = solve(problem, initial=initial, config=OptimizerConfig(seed=0))
accepted = result.plan
decoded = decode(problem, accepted)
```

`Decision(path_index=0, release_offset=0.0, speed_scale=1.0)` is immutable.
`path_index` is **zero-based**, whereas the manuscript's path indices start at 1.
`Plan` is a dictionary containing exactly one decision per robot. The initial
default selects each task's first supplied candidate, with zero offset and unit
speed scale; ordering candidate paths is the caller's responsibility.

`Evaluation` exposes:

| Field | Meaning |
| --- | --- |
| `violation` | Exact endpoint-integrated capacity violation, Eq. (9) or (12) |
| `operational_cost` | Four-term operational objective, Eq. (10) |
| `objective` | Tuple `(violation, operational_cost)` for strict lexicographic comparison |
| `feasible` | Whether `violation == 0.0` |
| `participation` | Per-robot linear-overload participation scores, Eq. (16) |
| `overloads` | Positive-overload cell-time intervals and occupying robot IDs |
| `decoded` | Protected reservations, full selected path lengths, and completion times |

An `OverloadInterval` records `cell`, `start`, `end`, `robots` and `excess`.
Integration splits at all occupancy and capacity-change endpoints; it does not
sample a time grid. Floating-point arithmetic is used. Objective comparisons
have no feasibility epsilon, so an operational saving cannot compensate for a
positive increase in computed capacity violation.

`PlanningResult` exposes `plan`, `evaluation`, `history`, `termination`,
`evaluations` and `sweeps`. History entries record the initial incumbent and the
incumbent after each processed block, including an interrupted final block.
Each entry has `sweep`, `block`, `objective`, `evaluations`, and a plan snapshot.
The objective sequence is nonincreasing within that result's fixed planning
cycle. Different cycles use different states/capacities and are not comparable
under the same monotonicity guarantee.

Termination values are `feasible`, `unchanged`, `evaluation_limit`, `time_limit`
and `max_sweeps`. `sweeps` counts sweeps entered, including a budget-truncated
final sweep. A feasible initial plan is returned after its initial evaluation,
without initiating cost-only optimization.

## Online state alignment

Call the same `solve` function with `initial=previous_result.plan` and a
`PlanningCycle(time=now, states=states, capacity_overrides=windows)`. A cycle with
positive time requires the preceding complete plan and an explicit state for
every robot. Initial time `0` uses unreleased robots and base capacities.

`RobotState` has the following fields:

| Field | Input contract |
| --- | --- |
| `status` | `"unreleased"`, `"active"` or `"completed"` |
| `executed_prefix` | Active: origin through observed current cell, inclusive; completed: full executed path |
| `history` | Tuple of all realized protected `Reservation` intervals, including outstanding tails |
| `realized_release` | Actual release time for active/completed robots; `None` before release |
| `completion_time` | Actual completion time for completed robots only |

An unreleased state uses empty prefix/history. Its decision lower bound becomes
`max(0, now - scheduled_release)`; the previous offset is increased to this
bound if necessary. An empty admissible release interval is an error.

An active robot's release offset is aligned with its observed release and then
frozen. Its selected path must retain the entire executed prefix. The remaining
decode starts at `now` in the observed cell; a new speed scale affects only this
suffix. This is the paper's cell-level abstraction. A geometric mid-edge
position and remaining subcell distance are not represented by this API.

Completed robots retain their aligned decisions and observed completion times.
Their supplied protection tails still participate in capacity evaluation and
conflict grouping, although the optimizer never edits their decisions.

`Reservation(robot_id, cell, start, end)` represents a **protected** half-open
interval `[start, end)`: `end` already includes the protection time. Do not add
it twice. When forming history from an actually occupied nominal interval
`[a, b)`, supply `[a, b + protection_time)`. Do not clip the tail at `now`.
History must describe realized occupancy only and be common to every candidate
in a cycle. Its completeness is the observation provider's responsibility; the
planner cannot reconstruct missing history from a current cell alone. Repeated
protected uses of one cell by one robot are unioned before counting occupancy.

```python
from core import CapacityWindow, PlanningCycle

cycle = PlanningCycle(
    time=now,
    states=observed_states,
    capacity_overrides=(CapacityWindow(affected_cell, closure_start, closure_end, 0),),
)
updated = solve(problem, initial=result.plan, cycle=cycle)
```

Effective capacity windows are piecewise-constant overrides. They must be
nonoverlapping on each cell and lie inside the horizon; base capacity applies
outside them. Zero represents temporary unavailability. Known realized releases
outside the admissible decision bounds are rejected, not clipped. Providing
updated candidate paths is possible only if the previous indices and executed
prefixes remain meaningful; otherwise construct an explicitly aligned complete
plan before calling the solver.

## Budgets and implementation choices

`OptimizerConfig` defaults are:

| Parameter | Default |
| --- | --- |
| `population_size` | 30 |
| `block_evaluation_limit` | 150, including repair |
| `total_evaluation_limit` | 6000 |
| `max_sweeps` | 10 |
| `time_limit` | `None`: 120 seconds at time 0; 1 second after an update |
| `seed` | 0 |
| `repair_fraction` | 0.35 |

The initial complete evaluation is counted once. A block reuses its already
evaluated incumbent, and all newly evaluated evolutionary and repair candidates
consume its local and global allowance. Time is checked before each new
evaluation; an evaluation already running can finish, so the wall-clock limit
is cooperative rather than a hard real-time deadline. Results are kept even if
the final block is interrupted.

The seed, mutation distributions and repair-budget split are explicit
implementation choices. The repair correction sets and objective weights are
taken from Section IV-A. See [PAPER_ALIGNMENT.md](PAPER_ALIGNMENT.md) for the
complete formula mapping and scope limits.
