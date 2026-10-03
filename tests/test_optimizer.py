"""Planner invariants, with deterministic objective oracles where useful.

The oracle tests isolate Algorithm 1's acceptance/budget logic from occupancy
integration. The first tests also exercise the real decoder and evaluator.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from core.capacity import evaluate
from core.coevolution import OptimizerConfig, solve
from core.models import (
    Bounds, Decision, PlanningCycle, Problem, RobotState, RobotTask, Workspace,
)
from core.repair import capacity_guided_repair


def problem_with_robots(count=3):
    workspace = Workspace(
        {"a": 1, "b": 1, "c": 1, "d": 1, "u": 1, "v": 1},
        {("a", "b"): 5, ("b", "d"): 5,
         ("a", "c"): 6, ("c", "d"): 6, ("u", "v"): 5},
    )
    tasks = tuple(
        RobotTask(robot_id, (("a", "b", "d"), ("a", "c", "d")), 1.0)
        for robot_id in range(1, count + 1)
    )
    return Problem(workspace, tasks, 100.0, protection_time=1.0,
                   bounds=Bounds(max_delay=20.0))


def oracle_evaluation(violation=1.0, cost=1.0, participation=None):
    return SimpleNamespace(
        violation=violation, operational_cost=cost,
        objective=(violation, cost), feasible=(violation == 0.0),
        participation=participation or {1: 1.0, 2: 1.0}, overloads=(),
    )


class OptimizerTests(unittest.TestCase):
    def test_feasible_plan_stops_without_cost_polishing(self):
        problem = problem_with_robots(1)
        initial = {1: Decision(1, 12.0, 0.9)}
        with patch("core.coevolution.ordered_blocks") as graph:
            result = solve(problem, initial=initial)
        self.assertEqual(result.termination, "feasible")
        self.assertEqual(result.evaluations, 1)
        self.assertEqual(result.sweeps, 0)
        self.assertEqual(result.plan, initial)
        graph.assert_not_called()

    def test_real_objective_is_nonincreasing_within_cycle(self):
        problem = problem_with_robots(3)
        result = solve(problem, config=OptimizerConfig(
            seed=7, block_evaluation_limit=35, total_evaluation_limit=120,
            max_sweeps=3, time_limit=10.0,
        ))
        objectives = [entry.objective for entry in result.history]
        self.assertTrue(all(after <= before for before, after in zip(objectives, objectives[1:])))
        self.assertEqual(evaluate(problem, result.plan, PlanningCycle()).objective,
                         result.evaluation.objective)
        self.assertLessEqual(result.evaluations, 120)

    def test_total_budget_includes_initial_evaluation(self):
        problem = problem_with_robots(2)
        with patch("core.coevolution.evaluate", wraps=evaluate) as evaluation:
            result = solve(problem, config=OptimizerConfig(total_evaluation_limit=1))
        self.assertEqual(evaluation.call_count, 1)
        self.assertEqual(result.evaluations, 1)
        self.assertEqual(result.termination, "evaluation_limit")

    def test_zero_time_limit_still_evaluates_initial_incumbent(self):
        result = solve(problem_with_robots(2), config=OptimizerConfig(time_limit=0.0))
        self.assertEqual(result.evaluations, 1)
        self.assertEqual(result.termination, "time_limit")

    def test_online_cycle_requires_previous_accepted_plan(self):
        with self.assertRaisesRegex(ValueError, "preceding accepted plan"):
            solve(problem_with_robots(2), cycle=PlanningCycle(time=1.0))

    def test_explicit_unreleased_states_at_zero_use_initial_time_limit(self):
        problem = problem_with_robots(2)
        calls = iter([0.0])

        def clock():
            return next(calls, 2.0)

        cycle = PlanningCycle(states={1: RobotState(), 2: RobotState()})
        with patch("core.coevolution.monotonic", side_effect=clock):
            result = solve(problem, cycle=cycle, config=OptimizerConfig(
                block_evaluation_limit=1, total_evaluation_limit=2,
            ))
        self.assertEqual(result.evaluations, 2)
        self.assertNotEqual(result.termination, "time_limit")

    def test_exact_ties_retain_incumbent_and_stop_unchanged(self):
        problem = problem_with_robots(2)
        initial = {1: Decision(), 2: Decision()}
        with patch("core.coevolution.evaluate", return_value=oracle_evaluation()), \
             patch("core.coevolution.ordered_blocks", return_value=[(1, 2)]):
            result = solve(problem, initial, config=OptimizerConfig(
                block_evaluation_limit=20, max_sweeps=5, time_limit=10.0,
            ))
        self.assertEqual(result.plan, initial)
        self.assertEqual(result.termination, "unchanged")
        self.assertEqual(result.sweeps, 1)

    def test_tiny_violation_increase_cannot_buy_lower_cost(self):
        problem = problem_with_robots(2)
        initial = {1: Decision(), 2: Decision()}

        def objective(_problem, plan, _cycle):
            return oracle_evaluation(1.0, 100.0) if plan == initial else \
                oracle_evaluation(1.0 + 1e-12, 0.0)

        with patch("core.coevolution.evaluate", side_effect=objective), \
             patch("core.coevolution.ordered_blocks", return_value=[(1, 2)]):
            result = solve(problem, initial, config=OptimizerConfig(
                block_evaluation_limit=20, time_limit=10.0,
            ))
        self.assertEqual(result.plan, initial)
        self.assertEqual(result.evaluation.objective, (1.0, 100.0))

    def test_block_context_is_fixed_and_inactive_robots_are_frozen(self):
        problem = problem_with_robots(5)
        initial = {robot_id: Decision() for robot_id in range(1, 6)}
        evaluated = []

        def objective(_problem, plan, _cycle):
            evaluated.append(dict(plan))
            return oracle_evaluation(
                1000.0 - sum(item.release_offset for item in plan.values()),
                participation={robot_id: 1.0 for robot_id in plan},
            )

        with patch("core.coevolution.evaluate", side_effect=objective), \
             patch("core.coevolution.ordered_blocks", return_value=[(1, 2), (3, 4)]) as graph:
            result = solve(problem, initial, config=OptimizerConfig(
                block_evaluation_limit=12, max_sweeps=1, time_limit=10.0,
            ))
        self.assertEqual(graph.call_count, 1)
        first_block_end = result.history[1].evaluations
        first_block_plan = result.history[1].plan
        for candidate in evaluated[1:first_block_end]:
            self.assertEqual({i: candidate[i] for i in (3, 4, 5)},
                             {i: initial[i] for i in (3, 4, 5)})
        for candidate in evaluated[first_block_end:]:
            self.assertEqual({i: candidate[i] for i in (1, 2, 5)},
                             {i: first_block_plan[i] for i in (1, 2, 5)})
        self.assertEqual(result.plan[5], initial[5])
        self.assertEqual(len(evaluated), result.evaluations)
        self.assertLessEqual(result.evaluations, 25)  # Initial + two block budgets.

    def test_repair_gets_reserved_budget_and_all_calls_are_counted(self):
        problem = problem_with_robots(2)
        at_repair = []
        from core.repair import capacity_guided_repair as real_repair

        def repair(*args, **kwargs):
            at_repair.append(evaluation.call_count)
            return real_repair(*args, **kwargs)

        with patch("core.coevolution.evaluate", return_value=oracle_evaluation()) as evaluation, \
             patch("core.coevolution.ordered_blocks", return_value=[(1, 2)]), \
             patch("core.coevolution.capacity_guided_repair", side_effect=repair):
            result = solve(problem, config=OptimizerConfig(
                block_evaluation_limit=10, total_evaluation_limit=11,
                repair_fraction=0.35, time_limit=10.0,
            ))
        self.assertTrue(at_repair)
        self.assertLessEqual(at_repair[0], 7)  # Initial + at most 6 evolution calls.
        self.assertEqual(evaluation.call_count, result.evaluations)
        self.assertEqual(result.evaluations, 11)

    def test_online_candidates_preserve_release_prefix_and_completed_decisions(self):
        problem = problem_with_robots(3)
        initial = {1: Decision(0, 0.0, 1.0), 2: Decision(1, 2.0, 0.9), 3: Decision(0, 5.0, 1.0)}
        cycle = PlanningCycle(time=5.0, states={
            1: RobotState("active", ("a", "b"), realized_release=0.0),
            2: RobotState("completed", ("a", "c", "d"), realized_release=2.0,
                          completion_time=4.0),
        })
        candidates = []

        def objective(_problem, plan, _cycle):
            candidates.append(dict(plan))
            return oracle_evaluation(participation={1: 3.0, 2: 2.0, 3: 1.0})

        # Alignment is separately tested in test_decoding/online; this oracle
        # isolates whether local operators respect an already aligned context.
        with patch("core.coevolution.align_plan", side_effect=lambda p, x, c: dict(x)), \
             patch("core.coevolution.evaluate", side_effect=objective), \
             patch("core.coevolution.ordered_blocks", return_value=[(1, 2, 3)]):
            solve(problem, initial, cycle, OptimizerConfig(
                block_evaluation_limit=30, time_limit=10.0,
            ))
        self.assertGreater(len(candidates), 1)
        for candidate in candidates:
            self.assertEqual(candidate[1].release_offset, 0.0)
            self.assertEqual(candidate[1].path_index, 0)
            self.assertEqual(candidate[2], initial[2])
            self.assertGreaterEqual(candidate[3].release_offset, 5.0)

    def test_repair_candidates_change_only_one_component(self):
        problem = problem_with_robots(2)
        initial = {1: Decision(), 2: Decision()}
        candidates = []
        evaluation = oracle_evaluation(participation={1: 4.0, 2: 2.0})

        def objective(plan):
            candidates.append(dict(plan))
            return evaluation

        plan, _ = capacity_guided_repair(
            problem, initial, evaluation, (1, 2), PlanningCycle(), objective,
        )
        self.assertEqual(plan, initial)
        self.assertTrue(candidates)
        changed_robots = []
        for candidate in candidates:
            changed = [i for i in initial if candidate[i] != initial[i]]
            self.assertEqual(len(changed), 1)
            changed_robots.extend(changed)
            one, two = initial[changed[0]], candidate[changed[0]]
            self.assertEqual(sum((one.path_index != two.path_index,
                                  one.release_offset != two.release_offset,
                                  one.speed_scale != two.speed_scale)), 1)
        self.assertEqual(changed_robots, sorted(changed_robots))

    def test_repair_recomputes_participation_after_accepted_correction(self):
        problem = problem_with_robots(2)
        initial = {1: Decision(), 2: Decision()}
        candidates = []

        def objective(plan):
            candidates.append(dict(plan))
            if plan[1].release_offset > 0:
                return oracle_evaluation(0.5, participation={1: 0.5, 2: 0.0})
            return oracle_evaluation(1.0, participation={1: 4.0, 2: 2.0})

        plan, evaluation = capacity_guided_repair(
            problem, initial, oracle_evaluation(participation={1: 4.0, 2: 2.0}),
            (1, 2), PlanningCycle(), objective,
        )
        self.assertLess(evaluation.violation, 1.0)
        self.assertGreater(plan[1].release_offset, 0.0)
        self.assertTrue(all(candidate[2] == initial[2] for candidate in candidates))

    def test_repair_compares_examined_robots_complete_candidate_families(self):
        problem = problem_with_robots(2)
        initial = {1: Decision(), 2: Decision()}
        candidates = []

        def objective(plan):
            candidates.append(dict(plan))
            if plan[1].release_offset == 2.5:
                return oracle_evaluation(0.0, 10.0)
            if plan[1].release_offset == 5.0:
                return oracle_evaluation(0.0, 1.0)
            return oracle_evaluation(participation={1: 4.0, 2: 2.0})

        plan, evaluation = capacity_guided_repair(
            problem, initial, oracle_evaluation(participation={1: 4.0, 2: 2.0}),
            (1, 2), PlanningCycle(), objective,
        )
        self.assertEqual(evaluation.objective, (0.0, 1.0))
        self.assertEqual(plan[1].release_offset, 5.0)
        self.assertTrue(any(candidate[1].speed_scale != 1.0 for candidate in candidates))
        self.assertTrue(all(candidate[2] == initial[2] for candidate in candidates))


if __name__ == "__main__":
    unittest.main()
