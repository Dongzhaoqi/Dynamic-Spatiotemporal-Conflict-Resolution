"""Integrated conflict-structured coevolution (Algorithm 1).

The manuscript specifies mixed-variable operators, not their exact stochastic
distribution. This implementation uses discrete uniform crossover/path mutation
and bounded Gaussian mutation of the two continuous variables. It is not an
implementation of differential evolution or a claim to reproduce experiment
trajectories. See ``OptimizerConfig`` for explicitly chosen search details.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, isfinite
from random import Random
from time import monotonic

from .capacity import Evaluation, evaluate
from .conflict_graph import ordered_blocks
from .decoding import admissible_path_indices, release_lower_bound
from .models import Decision, Plan, PlanningCycle, Problem
from .online import align_plan
from .repair import capacity_guided_repair


@dataclass(frozen=True)
class OptimizerConfig:
    """Table I budgets plus explicit, configurable implementation choices.

    ``block_evaluation_limit`` includes BOTH evolution and repair. The incumbent
    is reused without another evaluation. Up to ``repair_fraction`` (rounded up,
    at least one) of the available block budget is reserved for repair; repair
    also receives any evolution budget left unused. The default 35% reservation
    and random seed are implementation choices absent from the manuscript.

    ``time_limit=None`` means 120 seconds for an initial cycle, or 1 second when
    the update time is positive. Explicit unreleased states at time zero still
    describe an initial cycle.
    The initial complete evaluation is always performed and counted, even when
    the time limit is zero. Time is checked before each subsequent evaluation;
    an evaluation already in progress is allowed to finish.
    """

    population_size: int = 30
    block_evaluation_limit: int = 150
    total_evaluation_limit: int = 6000
    max_sweeps: int = 10
    time_limit: float | None = None
    seed: int = 0
    repair_fraction: float = 0.35

    def __post_init__(self):
        for name in ("population_size", "block_evaluation_limit", "total_evaluation_limit"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if type(self.max_sweeps) is not int or self.max_sweeps < 0:
            raise ValueError("max_sweeps must be a nonnegative integer.")
        if self.time_limit is not None and (
            not isfinite(self.time_limit) or self.time_limit < 0
        ):
            raise ValueError("time_limit must be finite and nonnegative.")
        if not isfinite(self.repair_fraction) or not 0 < self.repair_fraction <= 1:
            raise ValueError("repair_fraction must be in (0, 1].")


@dataclass(frozen=True)
class HistoryEntry:
    """Initial or accepted block incumbent, comparable only within this cycle."""

    sweep: int
    block: tuple[int, ...]
    objective: tuple[float, float]
    evaluations: int
    plan: Plan


@dataclass(frozen=True)
class PlanningResult:
    plan: Plan
    evaluation: Evaluation
    history: tuple[HistoryEntry, ...]
    termination: str
    evaluations: int
    # Number of sweeps entered; a budget-truncated final sweep is included.
    sweeps: int


def _offspring(
    problem: Problem,
    cycle: PlanningCycle,
    block: tuple[int, ...],
    first: Plan,
    second: Plan,
    rng: Random,
) -> Plan:
    """Generate an admissible block; completed robots are omitted by caller."""
    result = {}
    for robot_id in block:
        one, two = first[robot_id], second[robot_id]
        paths = admissible_path_indices(problem, robot_id, cycle)
        index = one.path_index if rng.random() < 0.5 else two.path_index
        if rng.random() < 0.35:
            index = rng.choice(paths)
        if cycle.state_for(robot_id).status == "unreleased":
            lower = release_lower_bound(problem, robot_id, cycle)
            blend = rng.random()
            delay = blend * one.release_offset + (1.0 - blend) * two.release_offset
            if rng.random() < 0.8:
                delay += rng.gauss(0.0, 0.10 * (problem.bounds.max_delay - lower))
            delay = min(problem.bounds.max_delay, max(lower, delay))
        else:
            # All population members inherit the same realized release time.
            delay = one.release_offset
        blend = rng.random()
        speed = blend * one.speed_scale + (1.0 - blend) * two.speed_scale
        if rng.random() < 0.8:
            speed += rng.gauss(0.0, 0.125 * (
                problem.bounds.speed_max - problem.bounds.speed_min
            ))
        speed = min(problem.bounds.speed_max, max(problem.bounds.speed_min, speed))
        result[robot_id] = Decision(index, delay, speed)
    return result


def solve(
    problem: Problem,
    initial: Plan | None = None,
    cycle: PlanningCycle | None = None,
    config: OptimizerConfig | None = None,
) -> PlanningResult:
    """Return a nonworsening incumbent using fixed-context sequential blocks.

    ``initial`` is a full decision dictionary, and must be the preceding cycle's
    accepted plan for an online cycle. Omitting it in the initial cycle
    initializes scheduled decisions. ``align_plan``
    enforces the observed-state domain before the first evaluation. Termination
    is one of ``feasible``, ``unchanged``, ``evaluation_limit``, ``time_limit`` or
    ``max_sweeps``. No comparison across distinct planning cycles is asserted.

    History contains the initial incumbent and each block's best incumbent,
    including an unfinished block when a computation budget is exhausted. A
    complete candidate is retained only on a strict tuple comparison; no epsilon
    tolerance can trade increased capacity violation against operational cost.
    """
    config = config or OptimizerConfig()
    cycle = cycle or PlanningCycle()
    started = monotonic()
    online = cycle.time > 0
    if online and initial is None:
        raise ValueError("Online cycles require the preceding accepted plan as initial.")
    limit = config.time_limit if config.time_limit is not None else (1.0 if online else 120.0)
    deadline = started + limit
    rng = Random(config.seed)
    scheduled = {task.robot_id: Decision() for task in problem.tasks}
    incumbent = align_plan(problem, scheduled if initial is None else dict(initial), cycle)
    incumbent_evaluation = evaluate(problem, incumbent, cycle)
    evaluation_count = 1
    history = [HistoryEntry(0, (), incumbent_evaluation.objective, 1, dict(incumbent))]
    sweeps = 0

    def finish(reason: str) -> PlanningResult:
        return PlanningResult(
            dict(incumbent), incumbent_evaluation, tuple(history), reason,
            evaluation_count, sweeps,
        )

    def exhausted() -> str | None:
        if evaluation_count >= config.total_evaluation_limit:
            return "evaluation_limit"
        if monotonic() >= deadline:
            return "time_limit"
        return None

    if incumbent_evaluation.feasible:
        return finish("feasible")
    while sweeps < config.max_sweeps:
        reason = exhausted()
        if reason:
            return finish(reason)
        sweeps += 1
        sweep_start = dict(incumbent)
        # Eq. 14 partition and Eq. 16 ordering stay fixed throughout this sweep.
        blocks = ordered_blocks(incumbent_evaluation)
        for graph_block in blocks:
            reason = exhausted()
            if reason:
                return finish(reason)
            block = tuple(
                robot_id for robot_id in graph_block
                if cycle.state_for(robot_id).status != "completed"
            )
            if not block:
                continue
            context = dict(incumbent)
            best_plan, best_evaluation = dict(incumbent), incumbent_evaluation
            block_evaluations = 0
            allowance = min(
                config.block_evaluation_limit,
                config.total_evaluation_limit - evaluation_count,
            )
            # Reserve at least one evaluation, even for a one-evaluation block.
            repair_allowance = max(1, ceil(allowance * config.repair_fraction))
            evolution_allowance = allowance - repair_allowance

            def evaluate_trial(candidate: Plan) -> Evaluation | None:
                nonlocal evaluation_count, block_evaluations, best_plan, best_evaluation
                if block_evaluations >= allowance or exhausted():
                    return None
                result = evaluate(problem, candidate, cycle)
                evaluation_count += 1
                block_evaluations += 1
                # Every evaluated trial participates in Eq. 17, even if it does
                # not survive the local population selection or repair pass.
                if result.objective < best_evaluation.objective:
                    best_plan, best_evaluation = dict(candidate), result
                return result

            entering_block = {robot_id: context[robot_id] for robot_id in block}
            population = [(entering_block, incumbent_evaluation)]
            attempts = 0
            # Finite proposal attempts also handle a completely fixed domain.
            max_attempts = max(16, evolution_allowance * 4)
            while block_evaluations < evolution_allowance and attempts < max_attempts:
                if exhausted() or best_evaluation.feasible:
                    break
                attempts += 1
                if len(population) < config.population_size:
                    first = second = entering_block
                else:
                    first = rng.choice(population)[0]
                    second = rng.choice(population)[0]
                trial_block = _offspring(problem, cycle, block, first, second, rng)
                if trial_block == entering_block:
                    continue
                candidate = dict(context)
                candidate.update(trial_block)
                trial_evaluation = evaluate_trial(candidate)
                if trial_evaluation is None:
                    break
                population.append((trial_block, trial_evaluation))
                population.sort(key=lambda item: item[1].objective)
                del population[config.population_size:]
            if not best_evaluation.feasible and not exhausted():
                # Repair changes only the block; every other robot still has
                # exactly its value in the entering full-plan context.
                capacity_guided_repair(
                    problem, best_plan, best_evaluation, block, cycle, evaluate_trial,
                )
            incumbent, incumbent_evaluation = best_plan, best_evaluation
            history.append(HistoryEntry(
                sweeps, block, incumbent_evaluation.objective,
                evaluation_count, dict(incumbent),
            ))
            if incumbent_evaluation.feasible:
                return finish("feasible")
            reason = exhausted()
            if reason:
                return finish(reason)
        if incumbent == sweep_start:
            return finish("unchanged")
    return finish("max_sweeps")
