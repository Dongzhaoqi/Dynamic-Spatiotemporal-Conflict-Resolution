# Alignment with the manuscript

本目录聚焦论文算法。模型、目标函数、冲突分块、修复和在线更新按论文第 II–III 节整理；实验图、场景生成、对比算法及地图和实机流程均不包含在发布目录中。

The reference is the supplied manuscript, **Dynamic Spatiotemporal Conflict Resolution for Multi–Robot Path Planning in Three–Dimensional Workspaces**. This document maps the implementation to Sections II–III, Eqs. (1)–(19), and Algorithm 1. Algorithm defaults also use the explicit settings in Section IV-A and Table I. It describes a substantial algorithm refactor, not a claim that the experimental results have been reproduced.

## Equation and module map

| Manuscript | Required meaning | Implementation |
| --- | --- | --- |
| Eq. (1) | Transition time is positive edge length divided by traversal speed. | `core/models.py`: `Workspace.distances`; `core/decoding.py`: `decode` |
| Eq. (2) | Each robot selects from a finite set of valid cell paths with common origin and destination. | `core/models.py`: `RobotTask.paths`, `Problem` validation |
| Eq. (3) | The decision consists of path index, release offset, and speed scale within their bounds. | `core/models.py`: `Decision`, `Bounds`; `core/decoding.py`: admissibility validation |
| Eqs. (4)–(5) | Recursively decode contiguous motion intervals on the source cells of path transitions. | `core/decoding.py`: `decode` |
| Eqs. (6)–(7) | Extend each occupancy at its end by the protection time; one robot contributes at most one occupancy to a cell at any time. | `core/models.py`: `Reservation`; `core/capacity.py`: protected interval union and occupancy sweep |
| Eqs. (8)–(9) | Capacity feasibility and the time integral of squared positive capacity excess. | `core/capacity.py`: `evaluate` |
| Eq. (10) | Sum release cost, selected path length cost, squared speed deviation cost, and makespan cost. | `core/models.py`: `ObjectiveWeights`; `core/capacity.py`: operational objective |
| Eq. (11) | Compare complete plans lexicographically by `(V_cap, J_op)`. | `Evaluation.objective` and candidate selection in `core/coevolution.py` / `core/repair.py` |
| Eq. (12) | Evaluate only the remaining horizon with the cycle's fixed effective capacity map. | `PlanningCycle`, `CapacityWindow`; `core/capacity.py` |
| Eq. (13) | Identify positive overload intervals. | `Evaluation.overloads` |
| Eq. (14) | Connect robots only when they share an overloaded cell–time resource; blocks are connected components. | `core/conflict_graph.py`: `ordered_blocks` |
| Eq. (15) | Replace only the current block's decisions and evaluate the resulting complete plan. | `core/coevolution.py` |
| Eq. (16) | Score each robot by its time integral of participation in positive overload, using the first power of the excess. | `Evaluation.participation`; block ordering and `core/repair.py` |
| Eq. (17) | Select from the incumbent and all evaluated block candidates, including repair candidates; preserve the incumbent on exact ties. | `core/coevolution.py` |
| Eq. (18) | A sweep does not worsen the complete plan's objective pair within a fixed planning cycle. | Sequential incumbent acceptance in `core/coevolution.py` |
| Eq. (19), Algorithm 1 | Return the accepted complete decision vector and update only unexecuted decisions. | `core/coevolution.py`: `solve`; `core/online.py`: `align_plan` |

## Model and decoder

The workspace is a generic directed cell graph. `Workspace.capacities` provides a positive integer base capacity for every cell, and `Workspace.distances` provides a positive length for every permitted transition. The core does not infer geometric distances or insert missing transitions. Consistent distance, time, and speed units are the caller's responsibility.

Candidate paths are inputs. The manuscript assumes they are precomputed, so the core does not require a grid, a scenario name, a map, or a particular graph-search implementation. The implementation validates their cells, transitions, and common endpoints. A path must contain at least one transition.

The paper uses path indices `1, ..., K_i`; Python uses `0, ..., K_i - 1`. Robot identifiers are positive integers, so the paper's minimum-index tie rule has a direct interpretation.

For an initial-cycle decision, the decoder uses

```text
release_i = scheduled_release_i + release_offset_i
speed_i   = nominal_speed_i * speed_scale_i
next_time = current_time + edge_length / speed_i
```

For a path with `L` transitions, it creates `L` nominal motion intervals, one on each transition's source cell. A protected reservation is `[entry, next_time + protection_time)`. It does not insert intermediate waiting or an automatic final-cell dwell. Completion is arrival at the final cell, before the protection tail expires.

This follows Eq. (5). Prerelease staging and persistent destination occupancy are outside the decoded motion model, as stated immediately after that equation. Applications that consume capacity for those activities need an explicit reservation model for them.

## Capacity and conflict structure

For each cell and time, the implementation evaluates

```text
n(c, t)   = number of distinct robots with a protected occupancy at (c, t)
rho(c, t) = max(n(c, t) - effective_capacity(c, t), 0)
V_cap     = sum over cells of integral rho(c, t)^2 dt
q_i       = sum over cells of integral chi_i(c, t) * rho(c, t) dt
```

The integration starts at the current cycle time and ends at the planning horizon. All interval and capacity-change endpoints partition that horizon into intervals with constant occupancy and capacity. Summing each interval's duration times the required integrand evaluates these quantities without selecting a sampling time step. Arithmetic still uses floating-point numbers.

Overlapping protected intervals for the same robot and cell are treated as a union. They must not count the robot twice. Half-open intervals also ensure that an occupancy ending exactly when another begins does not overlap it.

Pairwise overlap is not itself a conflict. For example, two robots may overlap in a cell of capacity two without producing overload or a graph edge. When three overlap in that cell, all three become active graph vertices and all three pairs are connected for that overloaded interval. A single robot in a temporarily unavailable cell of capacity zero forms an active singleton component, even though it creates no pairwise edge.

Connected components are ordered at the start of a sweep by descending sum of their robots' `q_i` values, then by the smallest robot identifier. The components and their order remain fixed for that sweep. The next sweep reconstructs them from the updated complete plan. No community detection, maximum component size, or artificial component splitting is used.

Completed robots remain in the capacity evaluation when their realized protection tails extend into the remaining horizon. Their decisions are frozen. A component's adjustable members exclude completed robots; a component with no adjustable decision cannot be resolved by changing its completed members.

## Objective, local evolution, and repair

The operational objective is

```text
J_op = sum_i (
    lambda_delay * release_offset_i
    + lambda_path * selected_path_length_i
    + lambda_speed * (speed_scale_i - 1)^2
) + lambda_makespan * max_i completion_time_i
```

Every candidate is embedded in the complete incumbent before decoding and evaluation. A local population's objective is therefore the complete plan's objective, not the sum of independently evaluated local conflicts. Capacity violation has priority over operational cost; the two are not collapsed into a large-penalty scalar fitness.

The current block decision is included in the local population. Evolution changes discrete path indices and continuous release offsets and speed scales while respecting the current cycle's admissible decisions. The manuscript specifies mixed-variable evolutionary operators but does not fully specify a unique mutation distribution, crossover schedule, or random initialization procedure. This implementation uses discrete uniform crossover and path mutation, blending and bounded Gaussian mutation for continuous components, and a fixed configurable random seed. These concrete operators in `core/coevolution.py` are implementation choices; they should not be interpreted as additional equations from the paper.

The `150`-evaluation block budget includes both evolution and repair. The current incumbent's evaluation is reused. By default, `35%` of the available block evaluations, rounded upward with at least one evaluation, is reserved for repair; repair also receives any unused evolution allowance. This allocation is a configurable implementation choice, not a ratio reported by the manuscript. Every completed trial evaluation counts toward the run budget, including repair trials. The initial complete evaluation is also counted and is always performed, even if a zero time limit is requested. Time limits are checked before starting subsequent evaluations; an evaluation already in progress is allowed to finish.

Capacity-guided repair examines adjustable robots with positive `q_i`. For an examined robot, it evaluates three families on separate copies of the current complete candidate:

1. Each admissible alternative path index, with timing components unchanged.
2. Each prescribed release-offset correction, for an unreleased robot only, clipped to its current admissible release interval.
3. Each prescribed speed-scale correction, clipped to the speed bounds, with path and release unchanged.

Each repair candidate changes one decision component. The best lexicographic improvement is retained; unchanged candidates remain eligible. After an accepted correction, occupancy and participation are recomputed before the next robot is selected. Search budgets may stop enumeration before every theoretical candidate is evaluated; acceptance then uses the candidates actually evaluated under that budget.

The incumbent is eligible in every block update, and exact objective ties preserve it. Consequently, accepted block updates and completed sweeps cannot worsen the objective pair for a fixed cycle. This is a finite-budget search property, not a global optimality, completeness, or guaranteed feasibility claim. The solver can return an incumbent with residual capacity violation when no available update improves it or a budget is exhausted.

The planning cycle stops on capacity feasibility, an unchanged entire incumbent after a sweep, the sweep limit, or a computation/evaluation budget. It does not run a separate operational-cost improvement stage after reaching capacity feasibility. Pair refinement, ejection chains, cross-component coordination, and feasible-delay compression are not part of this release.

## State-aligned online updates

Online updates use the same capacity model, graph construction, block optimization, and acceptance rule. `core/online.py` aligns the preceding accepted plan with externally supplied observations; it does not create random disturbances or define a separate recovery region.

For a fixed `PlanningCycle`, observations, protected realized histories, and capacity overrides remain unchanged while candidates are evaluated. At a positive cycle time, the caller must provide an explicit observed state for every robot. The caller must provide the full known realized protected history; its completeness cannot be inferred from the current cell alone. The initial cycle uses unreleased states and the base capacity map, without capacity overrides. The three statuses have these meanings:

| Status | Fixed information | Adjustable decisions |
| --- | --- | --- |
| `unreleased` | Scheduled task and absence of realized motion | Path, release offset in `[max(0, cycle.time - scheduled_release), max_delay]`, and speed scale |
| `active` | Realized release, executed prefix through the observed cell, and realized occupancy history | A candidate path with the same executed prefix, and speed scale for remaining motion |
| `completed` | Complete executed path, realized release, completion time, occupancy history, and accepted decisions | None |

For an active robot, decoding begins at `cycle.time` in the observed cell, which is the final cell of `executed_prefix`. The prefix constraint is positional: a candidate must begin with the complete executed prefix, not merely contain the same observed cell somewhere else. The decoder joins the fixed realized protected reservations with future protected reservations. `Reservation.end` already includes the protection time, so historical reservations must not be extended a second time.

The paper uses observed cells and does not specify a continuous position or partially traversed edge model. This implementation follows that cell-level interpretation and does not infer fractional progress inside an edge. Applications requiring that detail need a corresponding state and transition-time model.

An observed realized release determines the aligned release offset relative to the original scheduled release. Realized quantities are not clipped to disguise an out-of-domain disturbance. Inputs that make the current admissible decision set empty or violate the specified decision domain are rejected explicitly.

The aligned Eq. (10) cost uses the selected complete path length, aligned release offset, current decision's scalar speed deviation, and aligned completion time. It does not integrate historical speed penalties over time. The manuscript fixes realized decisions but does not prescribe a separate historical speed-cost integral.

Effective capacities use nonoverlapping piecewise-constant `CapacityWindow` overrides; a capacity of zero represents temporary unavailability. Outside an override, the base cell capacity applies. Occupancy protection tails from earlier motion remain relevant after the update time, including tails belonging to completed robots.

The horizon must cover every admissible protected completion in the cycle. The core checks a conservative bound using the latest admissible release, longest admissible candidate, minimum speed, and protection time, together with fixed history endpoints. Increasing the horizon is preferable to silently discarding part of an admissible plan.

The nonworsening relation applies within one fixed cycle only. New observations or a changed capacity map can increase the next cycle's initial objective value, as the manuscript explicitly allows.

## Algorithm defaults from Section IV-A

| Parameter | Default |
| --- | --- |
| Maximum release offset | `60.0` |
| Speed scale bounds | `[0.80, 1.20]` |
| Protection time | `5.0` |
| Release correction set | `(-5.0, -2.5, 2.5, 5.0)` |
| Speed correction set | `(-0.10, -0.05, 0.05, 0.10)` |
| Operational weights: release / path / speed / makespan | `1.0 / 0.01 / 10.0 / 0.1` |
| Local population size | `30` |
| Maximum planning sweeps | `10` |
| Local evolution budget | `150` complete evaluations per block |
| Total evaluation limit | `6000` per run |
| Nominal computation limit | `120.0` seconds |

When `OptimizerConfig.time_limit` is omitted, the solver uses `120.0` seconds for the initial cycle and the paper's `1.0` second limit for a positive-time online cycle. Callers can override these limits. The paper also uses at most three candidate paths in the controlled experiments. The mathematical model permits a general finite `K_i`, so the core accepts the caller's finite candidate sets without embedding that experimental cap or any experimental workspace geometry.

## Scope of the release

Included: generic model inputs, motion decoding, state alignment, capacity evaluation, the induced conflict graph, mixed-variable block search, capacity-guided repair, and semantic checks.

Excluded: experimental datasets and result tables, random/bottleneck/confluence scene generation, plotting and figure layout, map processing, ROS/UWB integration, baseline algorithms, and claims of reproduced comparative performance. The code is a high-level cell-time planner under the paper's assumptions; it does not add a continuous robot-dynamics or low-level control model.
