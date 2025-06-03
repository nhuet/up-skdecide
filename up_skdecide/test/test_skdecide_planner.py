from unified_planning.shortcuts import *
from unified_planning.test.examples import get_example_problems

from skdecide.hub.domain.up import UPDomain as SkDecideDomain

from skdecide.hub.solver.iw import IW
from ray.rllib.algorithms.dqn import DQN
from skdecide.hub.solver.ray_rllib import RayRLlib


if "skdecide" not in get_environment().factory.engines:
    get_environment().factory.add_engine("skdecide", "up_skdecide", "EngineImpl")
problems = get_example_problems()


def test_domain_basic():
    problem = problems["basic"].problem
    domain = SkDecideDomain(problem)
    domain._get_next_state(
        domain._get_initial_state(), domain._get_action_space().sample()
    )


def test_planner_basic():
    problem = problems["basic"].problem
    plan = problems["basic"].valid_plans[0]
    with OneshotPlanner(
        name="skdecide",
        params={
            "solver": IW,
            "config": {"state_encoding": "vector", "state_features": lambda d, s: s},
        },
    ) as planner:
        assert planner is not None
        res = planner.solve(problem)
        assert str(plan) == str(res.plan)


def test_planner_rllib():
    problem = problems["basic_conditional"].problem
    plan = problems["basic_conditional"].valid_plans[0]
    with OneshotPlanner(
        name="skdecide",
        params={
            "solver": RayRLlib,
            "config": {"state_encoding": "vector", "action_encoding": "int", "algo_class": DQN, "train_iterations": 1}
        },
    ) as planner:
        assert planner is not None
        res = planner.solve(problem)
        assert str(plan) == str(res.plan)
