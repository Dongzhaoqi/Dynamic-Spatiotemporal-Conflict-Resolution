"""Capacity-guided single-component repair (Section III-B, Eq. 16).

Each tested correction changes exactly one component of one robot. Evaluation
is supplied by the caller so repair shares the local/global evaluation and time
budgets with evolution. A callback returning ``None`` means no budget remains.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable, TYPE_CHECKING

from .decoding import admissible_path_indices, release_lower_bound
from .models import Plan, PlanningCycle, Problem

if TYPE_CHECKING:
    from .capacity import Evaluation


DELAY_STEPS = (-5.0, -2.5, 2.5, 5.0)
SPEED_STEPS = (-0.10, -0.05, 0.05, 0.10)


def capacity_guided_repair(
    problem: Problem,
    plan: Plan,
    evaluation: Evaluation,
    block: tuple[int, ...],
    cycle: PlanningCycle,
    evaluate_candidate: Callable[[Plan], Evaluation | None],
    delay_steps: tuple[float, ...] = DELAY_STEPS,
    speed_steps: tuple[float, ...] = SPEED_STEPS,
) -> tuple[Plan, Evaluation]:
    """Visit positive-participation robots once, using updated scores each time.

    For a visited robot all three correction families originate from the same
    entering decision, rather than chaining component changes. The lexicographic
    best correction is accepted after these tests; exact ties retain the current
    candidate. Already visited robots are not revisited during this repair pass.
    After the examined robot's candidate families are compared, capacity
    feasibility ends the pass without visiting another robot or starting a
    separate cost-only search.
    """
    current = dict(plan)
    current_evaluation = evaluation
    unvisited = {
        robot_id for robot_id in block
        if cycle.state_for(robot_id).status != "completed"
    }
    while unvisited and not current_evaluation.feasible:
        positive = [
            robot_id for robot_id in unvisited
            if current_evaluation.participation.get(robot_id, 0.0) > 0.0
        ]
        if not positive:
            break
        robot_id = min(
            positive,
            key=lambda item: (-current_evaluation.participation[item], item),
        )
        unvisited.remove(robot_id)
        decision = current[robot_id]
        alternatives = [
            replace(decision, path_index=index)
            for index in admissible_path_indices(problem, robot_id, cycle)
            if index != decision.path_index
        ]
        if cycle.state_for(robot_id).status == "unreleased":
            lower = release_lower_bound(problem, robot_id, cycle)
            alternatives.extend(
                replace(
                    decision,
                    release_offset=min(
                        problem.bounds.max_delay,
                        max(lower, decision.release_offset + step),
                    ),
                )
                for step in delay_steps
            )
        alternatives.extend(
            replace(
                decision,
                speed_scale=min(
                    problem.bounds.speed_max,
                    max(problem.bounds.speed_min, decision.speed_scale + step),
                ),
            )
            for step in speed_steps
        )
        best_plan, best_evaluation = current, current_evaluation
        seen = {decision}
        for alternative in alternatives:
            # Clamping can duplicate the current decision or another correction.
            if alternative in seen:
                continue
            seen.add(alternative)
            candidate = dict(current)
            candidate[robot_id] = alternative
            candidate_evaluation = evaluate_candidate(candidate)
            if candidate_evaluation is None:
                return best_plan, best_evaluation
            if candidate_evaluation.objective < best_evaluation.objective:
                best_plan, best_evaluation = candidate, candidate_evaluation
        # Evaluations already contain recomputed occupancies and q_i values.
        current, current_evaluation = best_plan, best_evaluation
    return current, current_evaluation
