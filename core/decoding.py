"""Complete-plan decoding for Eqs. (1)-(7) and state alignment in Section III.

Future motion is continuous along each selected path: there is no intermediate
wait decision. A segment reserves its SOURCE cell, with a trailing protection
buffer. The destination does not receive an implicit dwell reservation.
"""

from __future__ import annotations

from math import isclose, isfinite
from typing import Mapping

from .models import Decision, DecodedPlan, PlanningCycle, Problem, Reservation


def admissible_path_indices(problem: Problem, robot_id: int, cycle: PlanningCycle) -> tuple[int, ...]:
    """Zero-based path indices preserving the entire realized prefix."""
    task = problem.task_map[robot_id]
    state = cycle.state_for(robot_id)
    if state.status == "unreleased":
        return tuple(range(len(task.paths)))
    prefix = state.executed_prefix
    if state.status == "completed":
        return tuple(k for k, path in enumerate(task.paths) if path == prefix)
    return tuple(k for k, path in enumerate(task.paths)
                 if len(path) > len(prefix) and path[:len(prefix)] == prefix)


def release_lower_bound(problem: Problem, robot_id: int, cycle: PlanningCycle) -> float:
    """The unreleased-robot lower bound in Section III-B."""
    return max(0.0, cycle.time - problem.task_map[robot_id].scheduled_release)


def validate_cycle(problem: Problem, cycle: PlanningCycle) -> None:
    """Reject inconsistent observed states instead of changing realized history.

The caller supplies the full realized protected history, including any tail
that extends past the update time. Completeness of sensor history is an input
contract; it cannot be recovered from a path and the current cell alone.
"""
    if not isfinite(cycle.time) or not 0 <= cycle.time <= problem.horizon:
        raise ValueError("Cycle time must lie inside the finite planning horizon.")
    ids = set(problem.task_map)
    if not set(cycle.states) <= ids:
        raise ValueError("Observed states contain unknown robot IDs.")
    if cycle.time > 0 and set(cycle.states) != ids:
        raise ValueError("Online cycles require an explicit state for every robot.")
    for task in problem.tasks:
        state = cycle.state_for(task.robot_id)
        if state.status not in {"unreleased", "active", "completed"}:
            raise ValueError("Unknown robot execution status.")
        if cycle.time == 0 and state.status != "unreleased":
            raise ValueError("At the initial cycle no decisions have been realized.")
        if state.status == "unreleased":
            if state.executed_prefix or state.history or state.realized_release is not None or state.completion_time is not None:
                raise ValueError("Unreleased robots cannot have a realized path or release.")
            if release_lower_bound(problem, task.robot_id, cycle) > problem.bounds.max_delay:
                raise ValueError("Unreleased robot has an empty admissible release interval.")
        else:
            if not state.executed_prefix or state.executed_prefix[0] != task.paths[0][0]:
                raise ValueError("Realized prefix must start at the task origin.")
            release = state.realized_release
            if release is None or not isfinite(release) or not 0 <= release <= cycle.time:
                raise ValueError("Active/completed robots require a valid realized release.")
            if not 0 <= release - task.scheduled_release <= problem.bounds.max_delay:
                raise ValueError("Realized release lies outside the paper's decision domain.")
            if not admissible_path_indices(problem, task.robot_id, cycle):
                raise ValueError("No supplied candidate path retains the realized prefix.")
            if state.status == "active" and state.completion_time is not None:
                raise ValueError("An active robot cannot have a completion time.")
            if state.status == "completed":
                end = state.completion_time
                if end is None or not isfinite(end) or not release <= end <= cycle.time:
                    raise ValueError("Completed robots require a realized completion time.")
        for reservation in state.history:
            if reservation.robot_id != task.robot_id or reservation.cell not in state.executed_prefix:
                raise ValueError("History must belong to the robot and its executed prefix.")
            if not all(isfinite(x) for x in (reservation.start, reservation.end)):
                raise ValueError("Historical interval endpoints must be finite.")
            if not 0 <= reservation.start < reservation.end <= problem.horizon:
                raise ValueError("History contains an invalid protected interval.")
            if reservation.start > cycle.time or reservation.end > cycle.time + problem.protection_time:
                raise ValueError("History may contain realized occupancy and its protection tail only.")
            if reservation.end - reservation.start < problem.protection_time:
                raise ValueError("Protected history must include its full trailing protection time.")
            if state.status == "completed" and reservation.end > state.completion_time + problem.protection_time:
                raise ValueError("Completed history cannot extend beyond completion plus protection.")
            if state.realized_release is not None and reservation.start < state.realized_release:
                raise ValueError("Realized motion cannot precede the realized release.")

    by_cell: dict[str, list] = {}
    for window in cycle.capacity_overrides:
        if window.cell not in problem.workspace.capacities:
            raise ValueError("Effective capacity refers to an unknown cell.")
        if type(window.capacity) is not int or window.capacity < 0:
            raise ValueError("Effective capacity must be a nonnegative integer.")
        if not all(isfinite(x) for x in (window.start, window.end)) or not 0 <= window.start < window.end <= problem.horizon:
            raise ValueError("Capacity windows must be nonempty intervals inside the horizon.")
        by_cell.setdefault(window.cell, []).append(window)
    for windows in by_cell.values():
        ordered = sorted(windows, key=lambda window: window.start)
        if any(a.end > b.start for a, b in zip(ordered, ordered[1:])):
            raise ValueError("Effective capacity windows on one cell must not overlap.")
    if cycle.time == 0 and cycle.capacity_overrides:
        raise ValueError("The initial cycle uses base capacities, as defined in Section III-A.")


def required_horizon(problem: Problem, cycle: PlanningCycle | None = None) -> float:
    """Conservative endpoint covering all admissible protected completions.

This guards against a falsely feasible plan obtained by silently truncating
late reservations at T. It is independent of the optimizer's sampled trials.
"""
    cycle = cycle or PlanningCycle()
    validate_cycle(problem, cycle)
    latest = cycle.time
    for task in problem.tasks:
        state = cycle.state_for(task.robot_id)
        if state.status == "completed":
            latest = max(latest, state.completion_time or 0.0)
        else:
            start = task.scheduled_release + problem.bounds.max_delay if state.status == "unreleased" else cycle.time
            first = 0 if state.status == "unreleased" else len(state.executed_prefix) - 1
            longest = max(problem.workspace.path_length(task.paths[k][first:])
                          for k in admissible_path_indices(problem, task.robot_id, cycle))
            latest = max(latest, start + longest / (task.nominal_speed * problem.bounds.speed_min) + problem.protection_time)
        latest = max(latest, max((r.end for r in state.history), default=0.0))
    return latest


def validate_plan(problem: Problem, plan: Mapping[int, Decision], cycle: PlanningCycle) -> None:
    if set(plan) != set(problem.task_map):
        raise ValueError("A complete plan must contain exactly one decision for every robot.")
    for task in problem.tasks:
        decision = plan[task.robot_id]
        if type(decision.path_index) is not int or decision.path_index not in admissible_path_indices(problem, task.robot_id, cycle):
            raise ValueError("Path index is invalid or changes a realized path prefix.")
        if not isfinite(decision.release_offset) or not 0 <= decision.release_offset <= problem.bounds.max_delay:
            raise ValueError("Release offset lies outside [0, Delta_max].")
        if not isfinite(decision.speed_scale) or not problem.bounds.speed_min <= decision.speed_scale <= problem.bounds.speed_max:
            raise ValueError("Speed scale lies outside [alpha_min, alpha_max].")
        state = cycle.state_for(task.robot_id)
        if state.status == "unreleased":
            if decision.release_offset < release_lower_bound(problem, task.robot_id, cycle):
                raise ValueError("An unreleased robot cannot be released before the update time.")
        elif not isclose(task.scheduled_release + decision.release_offset, state.realized_release, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("A realized release time cannot be changed by a candidate.")


def decode(problem: Problem, plan: Mapping[int, Decision], cycle: PlanningCycle | None = None) -> DecodedPlan:
    """Decode every robot, including fixed robots, in one shared state context."""
    cycle = cycle or PlanningCycle()
    needed = required_horizon(problem, cycle)
    if problem.horizon < needed:
        raise ValueError(f"Horizon {problem.horizon} is too short; require at least {needed}.")
    validate_plan(problem, plan, cycle)
    reservations: list[Reservation] = []
    lengths: dict[int, float] = {}
    completion: dict[int, float] = {}
    for task in problem.tasks:
        robot_id = task.robot_id
        decision = plan[robot_id]
        state = cycle.state_for(robot_id)
        path = task.paths[decision.path_index]
        lengths[robot_id] = problem.workspace.path_length(path)
        reservations.extend(state.history)
        if state.status == "completed":
            completion[robot_id] = float(state.completion_time)
            continue
        first = 0 if state.status == "unreleased" else len(state.executed_prefix) - 1
        time = task.scheduled_release + decision.release_offset if state.status == "unreleased" else cycle.time
        speed = task.nominal_speed * decision.speed_scale
        for a, b in zip(path[first:], path[first + 1:]):
            next_time = time + problem.workspace.distances[a, b] / speed
            reservations.append(Reservation(robot_id, a, time, next_time + problem.protection_time))
            time = next_time
        completion[robot_id] = time
    return DecodedPlan(tuple(reservations), lengths, completion)
