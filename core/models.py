"""Scene-independent inputs for the manuscript's cell-time planning model.

Robot IDs are positive integers; candidate path indices are zero-based in code.
Lengths, speeds and times must use a consistent unit system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType
from typing import Literal, Mapping


@dataclass(frozen=True)
class Workspace:
    capacities: Mapping[str, int]
    distances: Mapping[tuple[str, str], float]

    def __post_init__(self):
        object.__setattr__(self, "capacities", MappingProxyType(dict(self.capacities)))
        object.__setattr__(self, "distances", MappingProxyType(dict(self.distances)))
        if not self.capacities or any(type(c) is not int or c < 1 for c in self.capacities.values()):
            raise ValueError("Base cell capacities must be positive integers.")
        for (a, b), distance in self.distances.items():
            if a not in self.capacities or b not in self.capacities:
                raise ValueError("Every transition endpoint must be a workspace cell.")
            if not isfinite(distance) or distance <= 0:
                raise ValueError("Transition lengths must be finite and positive.")

    def path_length(self, path: tuple[str, ...]) -> float:
        return sum(self.distances[a, b] for a, b in zip(path, path[1:]))


@dataclass(frozen=True)
class RobotTask:
    robot_id: int
    paths: tuple[tuple[str, ...], ...]
    nominal_speed: float
    scheduled_release: float = 0.0

    def __post_init__(self):
        object.__setattr__(self, "paths", tuple(tuple(p) for p in self.paths))
        if type(self.robot_id) is not int or self.robot_id < 1:
            raise ValueError("Robot IDs must be positive integers.")
        if not self.paths or any(len(p) < 2 for p in self.paths):
            raise ValueError("Each task needs precomputed paths with at least one transition.")
        endpoints = {(p[0], p[-1]) for p in self.paths}
        if len(endpoints) != 1:
            raise ValueError("All candidate paths of a task must share origin and destination.")
        if not isfinite(self.nominal_speed) or self.nominal_speed <= 0:
            raise ValueError("Nominal speed must be finite and positive.")
        if not isfinite(self.scheduled_release) or self.scheduled_release < 0:
            raise ValueError("Scheduled release must be finite and nonnegative.")


@dataclass(frozen=True)
class Bounds:
    max_delay: float = 60.0
    speed_min: float = 0.80
    speed_max: float = 1.20

    def __post_init__(self):
        if not all(isfinite(x) for x in (self.max_delay, self.speed_min, self.speed_max)):
            raise ValueError("Decision bounds must be finite.")
        if self.max_delay < 0 or not 0 < self.speed_min <= 1 <= self.speed_max:
            raise ValueError("Require max_delay >= 0 and 0 < speed_min <= 1 <= speed_max.")


@dataclass(frozen=True)
class ObjectiveWeights:
    delay: float = 1.0
    path_length: float = 0.01
    speed: float = 10.0
    makespan: float = 0.1

    def __post_init__(self):
        if any(not isfinite(x) or x < 0 for x in (self.delay, self.path_length, self.speed, self.makespan)):
            raise ValueError("Operational weights must be finite and nonnegative.")


@dataclass(frozen=True)
class Problem:
    workspace: Workspace
    tasks: tuple[RobotTask, ...]
    horizon: float
    protection_time: float = 5.0
    bounds: Bounds = field(default_factory=Bounds)
    weights: ObjectiveWeights = field(default_factory=ObjectiveWeights)

    def __post_init__(self):
        object.__setattr__(self, "tasks", tuple(self.tasks))
        ids = [task.robot_id for task in self.tasks]
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("A problem needs tasks with unique robot IDs.")
        if not isfinite(self.horizon) or self.horizon <= 0:
            raise ValueError("Planning horizon must be finite and positive.")
        if not isfinite(self.protection_time) or self.protection_time < 0:
            raise ValueError("Protection time must be finite and nonnegative.")
        for task in self.tasks:
            for path in task.paths:
                if any(cell not in self.workspace.capacities for cell in path):
                    raise ValueError("Candidate path contains an unknown cell.")
                try:
                    self.workspace.path_length(path)
                except KeyError as error:
                    raise ValueError("Candidate path contains an infeasible transition.") from error

    @property
    def task_map(self) -> dict[int, RobotTask]:
        return {task.robot_id: task for task in self.tasks}


@dataclass(frozen=True)
class Decision:
    path_index: int = 0
    release_offset: float = 0.0
    speed_scale: float = 1.0


Plan = dict[int, Decision]


@dataclass(frozen=True)
class Reservation:
    """A protected half-open interval [start, end); end already includes delta."""
    robot_id: int
    cell: str
    start: float
    end: float


@dataclass(frozen=True)
class CapacityWindow:
    """Effective capacity override on [start, end); zero means unavailable."""
    cell: str
    start: float
    end: float
    capacity: int


@dataclass(frozen=True)
class RobotState:
    status: Literal["unreleased", "active", "completed"] = "unreleased"
    # Active: prefix includes the observed cell. Completed: entire executed path.
    executed_prefix: tuple[str, ...] = ()
    # Realized protected intervals, including tails extending beyond cycle.time.
    history: tuple[Reservation, ...] = ()
    realized_release: float | None = None
    completion_time: float | None = None

    def __post_init__(self):
        object.__setattr__(self, "executed_prefix", tuple(self.executed_prefix))
        object.__setattr__(self, "history", tuple(self.history))


@dataclass(frozen=True)
class PlanningCycle:
    time: float = 0.0
    states: Mapping[int, RobotState] = field(default_factory=dict)
    capacity_overrides: tuple[CapacityWindow, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "states", MappingProxyType(dict(self.states)))
        object.__setattr__(self, "capacity_overrides", tuple(self.capacity_overrides))

    def state_for(self, robot_id: int) -> RobotState:
        return self.states.get(robot_id, RobotState())


@dataclass(frozen=True)
class DecodedPlan:
    reservations: tuple[Reservation, ...]
    path_lengths: Mapping[int, float]
    completion_times: Mapping[int, float]
