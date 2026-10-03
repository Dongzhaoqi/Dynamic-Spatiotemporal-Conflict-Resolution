"""Exact cell-time capacity evaluation, Eqs. (7)--(12) and (16).

The endpoint sweep and per-robot occupancy union preserve the useful audit
semantics of the source project's ``experiments/metrics.py::audit_capacity``.
This standalone implementation also handles the remaining planning horizon,
time-dependent effective capacities, and overload participation. It has no
dependency on experiments, geometric collision proxies, or plotting code.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import fsum
from typing import Mapping

from .decoding import decode
from .models import Decision, DecodedPlan, PlanningCycle, Problem, Reservation


@dataclass(frozen=True)
class OverloadInterval:
    """An endpoint-defined half-open interval of positive capacity excess."""

    cell: str
    start: float
    end: float
    robots: frozenset[int]
    excess: int


@dataclass(frozen=True)
class Evaluation:
    violation: float
    operational_cost: float
    participation: dict[int, float]
    overloads: tuple[OverloadInterval, ...]
    decoded: DecodedPlan

    @property
    def objective(self) -> tuple[float, float]:
        """Capacity has strict lexicographic priority, as in Eq. (11)."""
        return self.violation, self.operational_cost

    @property
    def feasible(self) -> bool:
        return self.violation == 0.0


def _occupancy_union(
    reservations: tuple[Reservation, ...], start: float, end: float
) -> dict[str, list[tuple[int, float, float]]]:
    """Union each robot/cell's intervals so chi_i is binary, Eq. (7a).

    Protection tails from realized history are included whenever they reach
    the remaining horizon. Intervals that only touch have zero overlap.
    """
    grouped: dict[tuple[int, str], list[tuple[float, float]]] = defaultdict(list)
    for reservation in reservations:
        left = max(start, reservation.start)
        right = min(end, reservation.end)
        if left < right:
            grouped[reservation.robot_id, reservation.cell].append((left, right))

    by_cell: dict[str, list[tuple[int, float, float]]] = defaultdict(list)
    for (robot_id, cell), intervals in sorted(grouped.items()):
        ordered = sorted(intervals)
        left, right = ordered[0]
        for next_left, next_right in ordered[1:]:
            if next_left <= right:
                right = max(right, next_right)
            else:
                by_cell[cell].append((robot_id, left, right))
                left, right = next_left, next_right
        by_cell[cell].append((robot_id, left, right))
    return by_cell


def evaluate(
    problem: Problem,
    plan: Mapping[int, Decision],
    cycle: PlanningCycle = PlanningCycle(),
) -> Evaluation:
    """Decode the complete plan and integrate exact endpoint-constant loads.

    Decoding validates the plan, states, and nonoverlapping capacity windows.
    No temporal sampling is used. End events take effect before start events,
    which implements half-open protected occupancies and capacity overrides.
    The decoder determines occupancies; no destination dwell is added here.
    """
    decoded = decode(problem, plan, cycle)
    by_cell = _occupancy_union(decoded.reservations, cycle.time, problem.horizon)
    windows_by_cell = defaultdict(list)
    for window in cycle.capacity_overrides:
        if window.start < problem.horizon and window.end > cycle.time:
            windows_by_cell[window.cell].append(window)

    overloads: list[OverloadInterval] = []
    violation_terms: list[float] = []
    participation_terms: dict[int, list[float]] = {
        task.robot_id: [] for task in problem.tasks
    }

    for cell, intervals in sorted(by_cell.items()):
        arrivals: dict[float, set[int]] = defaultdict(set)
        departures: dict[float, set[int]] = defaultdict(set)
        for robot_id, left, right in intervals:
            arrivals[left].add(robot_id)
            departures[right].add(robot_id)

        capacity_starts: dict[float, int] = {}
        capacity_ends: set[float] = set()
        for window in windows_by_cell[cell]:
            capacity_starts[max(cycle.time, window.start)] = window.capacity
            capacity_ends.add(min(problem.horizon, window.end))

        endpoints = sorted(
            {cycle.time, problem.horizon}
            | set(arrivals)
            | set(departures)
            | set(capacity_starts)
            | capacity_ends
        )
        active: set[int] = set()
        base_capacity = problem.workspace.capacities[cell]
        capacity = base_capacity
        for left, right in zip(endpoints, endpoints[1:]):
            active.difference_update(departures.get(left, ()))
            active.update(arrivals.get(left, ()))
            if left in capacity_ends:
                capacity = base_capacity
            if left in capacity_starts:
                capacity = capacity_starts[left]
            excess = max(len(active) - capacity, 0)
            if not excess:
                continue

            duration = right - left
            overloads.append(
                OverloadInterval(cell, left, right, frozenset(active), excess)
            )
            violation_terms.append(duration * excess**2)
            for robot_id in active:
                participation_terms[robot_id].append(duration * excess)

    weights = problem.weights
    operational_cost = fsum(
        weights.delay * decision.release_offset
        + weights.path_length * decoded.path_lengths[robot_id]
        + weights.speed * (decision.speed_scale - 1.0) ** 2
        for robot_id, decision in plan.items()
    ) + weights.makespan * max(decoded.completion_times.values(), default=0.0)
    return Evaluation(
        violation=fsum(violation_terms),
        operational_cost=operational_cost,
        participation={robot_id: fsum(terms) for robot_id, terms in participation_terms.items()},
        overloads=tuple(overloads),
        decoded=decoded,
    )
