"""Analytical checks of manuscript capacity integrals and induced blocks."""

import unittest
from unittest.mock import patch

from core.capacity import Evaluation, OverloadInterval, evaluate
from core.conflict_graph import ordered_blocks
from core.models import (
    CapacityWindow,
    Decision,
    DecodedPlan,
    ObjectiveWeights,
    PlanningCycle,
    Problem,
    Reservation,
    RobotState,
    RobotTask,
    Workspace,
)


def simple_problem(count=2, capacity=1, duration=4.0, protection=0.0, horizon=100.0):
    return Problem(
        Workspace({"a": capacity, "b": capacity}, {("a", "b"): duration}),
        tuple(RobotTask(i, (("a", "b"),), 1.0) for i in range(1, count + 1)),
        horizon=horizon,
        protection_time=protection,
    )


class CapacityTests(unittest.TestCase):
    def test_overlaps_below_capacity_do_not_create_conflicts(self):
        problem = simple_problem(count=2, capacity=2)
        result = evaluate(problem, {1: Decision(), 2: Decision()})
        self.assertTrue(result.feasible)
        self.assertEqual(result.participation, {1: 0.0, 2: 0.0})
        self.assertEqual(ordered_blocks(result), [])

        problem = simple_problem(count=3, capacity=2)
        result = evaluate(problem, {i: Decision() for i in (1, 2, 3)})
        self.assertEqual(result.violation, 4.0)
        self.assertEqual(ordered_blocks(result), [(1, 2, 3)])

    def test_repeated_cell_protected_intervals_are_a_binary_union(self):
        problem = Problem(
            Workspace(
                {"a": 1, "b": 1, "c": 1},
                {("a", "b"): 1.0, ("b", "a"): 1.0, ("a", "c"): 1.0},
            ),
            (
                RobotTask(1, (("a", "b", "a", "c"),), 1.0),
                RobotTask(2, (("a", "c"),), 1.0),
            ),
            horizon=100.0,
            protection_time=4.0,
        )
        result = evaluate(problem, {1: Decision(), 2: Decision(release_offset=2.0)})
        # Robot 1's a intervals [0,5) and [2,7) represent one occupant.
        self.assertEqual(result.violation, 5.0)
        self.assertEqual(result.participation, {1: 5.0, 2: 5.0})
        self.assertEqual(result.overloads, (OverloadInterval("a", 2.0, 7.0, frozenset({1, 2}), 1),))

    def test_half_open_endpoints_do_not_overlap_or_add_destination_dwell(self):
        problem = simple_problem(duration=1.0)
        result = evaluate(problem, {1: Decision(), 2: Decision(release_offset=1.0)})
        self.assertEqual(result.violation, 0.0)
        self.assertEqual({interval.cell for interval in result.decoded.reservations}, {"a"})

    def test_capacity_zero_creates_an_active_singleton(self):
        problem = simple_problem(count=1)
        cycle = PlanningCycle(
            time=0.5,
            states={1: RobotState()},
            capacity_overrides=(CapacityWindow("a", 1.0, 3.0, 0),),
        )
        result = evaluate(problem, {1: Decision(release_offset=0.5)}, cycle)
        self.assertEqual(result.violation, 2.0)
        self.assertEqual(result.participation, {1: 2.0})
        self.assertEqual(ordered_blocks(result), [(1,)])

    def test_adjacent_capacity_windows_apply_start_at_the_shared_endpoint(self):
        problem = simple_problem(capacity=2)
        cycle = PlanningCycle(
            time=0.5,
            states={1: RobotState(), 2: RobotState()},
            capacity_overrides=(
                CapacityWindow("a", 1.0, 2.0, 0),
                CapacityWindow("a", 2.0, 3.0, 1),
            )
        )
        result = evaluate(problem, {1: Decision(release_offset=0.5), 2: Decision(release_offset=0.5)}, cycle)
        self.assertEqual(result.violation, 5.0)
        self.assertEqual(result.participation, {1: 3.0, 2: 3.0})
        self.assertEqual([(x.start, x.end, x.excess) for x in result.overloads], [(1.0, 2.0, 2), (2.0, 3.0, 1)])

    def test_participation_uses_excess_and_violation_uses_squared_excess(self):
        problem = simple_problem(count=4, capacity=1, duration=2.0)
        result = evaluate(problem, {i: Decision() for i in range(1, 5)})
        self.assertEqual(result.violation, 18.0)
        self.assertEqual(result.participation, {i: 6.0 for i in range(1, 5)})

    def test_completed_history_tail_still_occupies_remaining_horizon(self):
        problem = simple_problem(count=2, protection=4.0, horizon=100.0)
        cycle = PlanningCycle(
            time=5.0,
            states={
                1: RobotState(
                    status="completed",
                    executed_prefix=("a", "b"),
                    history=(Reservation(1, "a", 0.0, 8.0),),
                    realized_release=0.0,
                    completion_time=4.0,
                ),
                2: RobotState(),
            },
        )
        result = evaluate(problem, {1: Decision(), 2: Decision(release_offset=5.0)}, cycle)
        self.assertEqual(result.violation, 3.0)
        self.assertEqual(result.participation, {1: 3.0, 2: 3.0})
        self.assertEqual([(x.start, x.end) for x in result.overloads], [(5.0, 8.0)])
        self.assertEqual(ordered_blocks(result), [(1, 2)])

    def test_audit_clips_both_history_and_future_to_remaining_horizon(self):
        problem = simple_problem(count=2, horizon=10.0)
        decoded = DecodedPlan(
            (
                Reservation(1, "a", 0.0, 12.0),
                Reservation(2, "a", 1.0, 20.0),
            ),
            {1: 4.0, 2: 4.0},
            {1: 4.0, 2: 4.0},
        )
        # Isolate the integral from state alignment: only [5,10) contributes.
        with patch("core.capacity.decode", return_value=decoded):
            result = evaluate(problem, {1: Decision(), 2: Decision()}, PlanningCycle(time=5.0))
        self.assertEqual(result.violation, 5.0)
        self.assertEqual(result.participation, {1: 5.0, 2: 5.0})

    def test_operational_cost_and_strict_lexicographic_order(self):
        problem = Problem(
            Workspace({"a": 2, "b": 2}, {("a", "b"): 4.0}),
            (RobotTask(1, (("a", "b"),), 1.0),),
            horizon=100.0,
            protection_time=0.0,
            weights=ObjectiveWeights(delay=2.0, path_length=3.0, speed=5.0, makespan=7.0),
        )
        result = evaluate(problem, {1: Decision(release_offset=2.0, speed_scale=1.2)})
        expected = 2.0 * 2.0 + 3.0 * 4.0 + 5.0 * 0.2**2 + 7.0 * (2.0 + 4.0 / 1.2)
        self.assertAlmostEqual(result.operational_cost, expected)
        self.assertEqual(result.objective, (0.0, result.operational_cost))
        self.assertLess(result.objective, (1e-12, 0.0))


class ConflictGraphTests(unittest.TestCase):
    def test_transitive_components_order_by_sum_and_then_minimum_id(self):
        empty_decoded = DecodedPlan((), {}, {})
        result = Evaluation(
            1.0,
            0.0,
            {1: 2.0, 2: 4.0, 3: 2.0, 4: 4.0, 5: 4.0, 8: 9.0, 9: 0.0},
            (
                OverloadInterval("a", 0.0, 1.0, frozenset({1, 2}), 1),
                OverloadInterval("b", 0.0, 1.0, frozenset({2, 3}), 1),
                OverloadInterval("c", 0.0, 1.0, frozenset({4, 5}), 1),
                OverloadInterval("d", 0.0, 1.0, frozenset({8}), 1),
            ),
            empty_decoded,
        )
        self.assertEqual(ordered_blocks(result), [(8,), (1, 2, 3), (4, 5)])


if __name__ == "__main__":
    unittest.main()
