"""Capacity-induced graph components and sweep order, Eqs. (14) and (16)."""

from __future__ import annotations

from itertools import combinations
from math import fsum

from .capacity import Evaluation


def ordered_blocks(evaluation: Evaluation) -> list[tuple[int, ...]]:
    """Return active connected components in descending aggregate q_i order.

    Every overloaded cell-time interval induces a clique among its occupants.
    A lone robot in a cell of effective capacity zero is an active singleton.
    Ordinary overlaps at or below capacity produce no vertices or edges.
    Components with equal participation are ordered by minimum robot ID.

    Completed robots with still-protected history may appear in the graph.
    Candidate generation must keep their realized decisions fixed.
    """
    adjacency: dict[int, set[int]] = {}
    for interval in evaluation.overloads:
        for robot_id in interval.robots:
            adjacency.setdefault(robot_id, set())
        for first, second in combinations(sorted(interval.robots), 2):
            adjacency[first].add(second)
            adjacency[second].add(first)

    unseen = set(adjacency)
    blocks = []
    while unseen:
        stack = [min(unseen)]
        component: set[int] = set()
        while stack:
            robot_id = stack.pop()
            if robot_id not in unseen:
                continue
            unseen.remove(robot_id)
            component.add(robot_id)
            stack.extend(adjacency[robot_id] & unseen)
        blocks.append(tuple(sorted(component)))

    blocks.sort(
        key=lambda block: (
            -fsum(evaluation.participation[robot_id] for robot_id in block),
            block[0],
        )
    )
    return blocks
