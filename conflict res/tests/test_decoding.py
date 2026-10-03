"""Mathematical decoding and observed-state invariants; no experiment scenes."""

import unittest
from dataclasses import replace

from core import (
    Bounds, CapacityWindow, Decision, PlanningCycle, Problem, Reservation,
    RobotState, RobotTask, Workspace, align_plan, decode, evaluate,
)
from core.decoding import admissible_path_indices, required_horizon


def problem():
    graph = Workspace(
        {cell: 1 for cell in ("a", "b", "c", "d", "e")},
        {("a", "b"): 10.0, ("b", "c"): 10.0, ("b", "d"): 10.0,
         ("d", "c"): 10.0, ("a", "e"): 10.0, ("e", "c"): 10.0},
    )
    tasks = (RobotTask(1, (("a", "b", "c"), ("a", "b", "d", "c"), ("a", "e", "c")), 2.0),)
    return Problem(graph, tasks, horizon=100.0, protection_time=2.0, bounds=Bounds(10.0, 0.5, 2.0))


class DecodingTests(unittest.TestCase):
    def test_source_cells_trailing_buffer_and_no_terminal_dwell(self):
        decoded = decode(problem(), {1: Decision()})
        self.assertEqual(decoded.reservations, (
            Reservation(1, "a", 0.0, 7.0), Reservation(1, "b", 5.0, 12.0),
        ))
        self.assertEqual(decoded.completion_times[1], 10.0)
        self.assertEqual(decoded.path_lengths[1], 20.0)

    def test_horizon_must_cover_unsampled_admissible_alternatives(self):
        p = problem()
        self.assertEqual(required_horizon(p), 42.0)
        with self.assertRaisesRegex(ValueError, "too short"):
            decode(replace(p, horizon=20.0), {1: Decision()})

    def test_active_prefix_and_history_unchanged_under_suffix_speed(self):
        p = problem()
        history = (Reservation(1, "a", 0.0, 7.0),)
        cycle = PlanningCycle(5.0, {1: RobotState("active", ("a", "b"), history, 0.0)})
        self.assertEqual(admissible_path_indices(p, 1, cycle), (0, 1))
        decoded = decode(p, {1: Decision(1, 0.0, 2.0)}, cycle)
        self.assertEqual(decoded.reservations, history + (
            Reservation(1, "b", 5.0, 9.5), Reservation(1, "d", 7.5, 12.0),
        ))
        self.assertEqual(decoded.path_lengths[1], 30.0)
        self.assertEqual(decoded.completion_times[1], 10.0)
        with self.assertRaisesRegex(ValueError, "prefix"):
            decode(p, {1: Decision(2)}, cycle)
        with self.assertRaisesRegex(ValueError, "realized release"):
            decode(p, {1: Decision(0, 1.0)}, cycle)

    def test_realized_delay_is_aligned_once_then_frozen(self):
        p = problem()
        state = RobotState("active", ("a",), (), 2.0)
        cycle = PlanningCycle(3.0, {1: state})
        aligned = align_plan(p, {1: Decision()}, cycle)
        self.assertEqual(aligned[1].release_offset, 2.0)
        self.assertEqual(decode(p, aligned, cycle).reservations[0].start, 3.0)

    def test_unreleased_lower_bound(self):
        p = problem()
        cycle = PlanningCycle(4.0, {1: RobotState()})
        aligned = align_plan(p, {1: Decision()}, cycle)
        self.assertEqual(aligned[1], Decision(0, 4.0, 1.0))
        self.assertEqual(decode(p, aligned, cycle).reservations[0].start, 4.0)
        with self.assertRaisesRegex(ValueError, "empty admissible"):
            align_plan(p, aligned, PlanningCycle(11.0, {1: RobotState()}))

    def test_online_requires_complete_observations(self):
        with self.assertRaisesRegex(ValueError, "explicit state"):
            align_plan(problem(), {1: Decision()}, PlanningCycle(1.0))

    def test_completed_protection_tail_remains_in_remaining_horizon(self):
        p = problem()
        state = RobotState("completed", ("a", "b", "c"), (
            Reservation(1, "a", 0.0, 7.0), Reservation(1, "b", 5.0, 12.0),
        ), 0.0, 10.0)
        cycle = PlanningCycle(11.0, {1: state}, (CapacityWindow("b", 11.0, 13.0, 0),))
        result = evaluate(p, {1: Decision()}, cycle)
        self.assertEqual(result.violation, 1.0)
        self.assertEqual(result.participation[1], 1.0)
        self.assertEqual(result.decoded.completion_times[1], 10.0)

    def test_completed_history_cannot_extend_past_actual_completion_buffer(self):
        p = problem()
        state = RobotState("completed", ("a", "b", "c"), (Reservation(1, "b", 5.0, 20.0),), 0.0, 10.0)
        with self.assertRaisesRegex(ValueError, "completion plus protection"):
            decode(p, {1: Decision()}, PlanningCycle(20.0, {1: state}))

    def test_overlapping_capacity_definitions_rejected(self):
        windows = (CapacityWindow("b", 4.0, 8.0, 0), CapacityWindow("b", 7.0, 9.0, 1))
        with self.assertRaisesRegex(ValueError, "must not overlap"):
            decode(problem(), {1: Decision(0, 1.0)}, PlanningCycle(1.0, {1: RobotState()}, windows))

    def test_invalid_transition_and_incomplete_plan_rejected(self):
        p = problem()
        with self.assertRaisesRegex(ValueError, "infeasible transition"):
            replace(p, tasks=(RobotTask(1, (("a", "c"),), 2.0),))
        with self.assertRaisesRegex(ValueError, "exactly one decision"):
            decode(p, {})


if __name__ == "__main__":
    unittest.main()
