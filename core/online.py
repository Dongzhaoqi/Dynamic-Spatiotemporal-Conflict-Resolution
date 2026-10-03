"""State alignment shared by nominal and subsequent planning cycles.

There is no separate recovery-region or pairwise quick-repair algorithm.
Call coevolution.solve again with a new PlanningCycle and the accepted plan.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Mapping

from .decoding import release_lower_bound, validate_cycle, validate_plan
from .models import Decision, Plan, PlanningCycle, Problem


def align_plan(problem: Problem, previous: Mapping[int, Decision], cycle: PlanningCycle) -> Plan:
    """Fix observed releases and align unreleased offsets before Algorithm 1.

    Path choices are retained, and any inconsistent executed prefix is rejected.
    Active/completed offsets reflect the *observed* release, even if it differs
    from the previously planned release. The optimizer cannot change it again.
    """
    validate_cycle(problem, cycle)
    if set(previous) != set(problem.task_map):
        raise ValueError("State alignment requires the previous complete plan.")
    aligned = dict(previous)
    for task in problem.tasks:
        decision = previous[task.robot_id]
        state = cycle.state_for(task.robot_id)
        if state.status == "unreleased":
            offset = max(decision.release_offset, release_lower_bound(problem, task.robot_id, cycle))
        else:
            offset = float(state.realized_release) - task.scheduled_release
        aligned[task.robot_id] = replace(decision, release_offset=offset)
    validate_plan(problem, aligned, cycle)
    return aligned
