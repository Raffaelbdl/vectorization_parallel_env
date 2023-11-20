import numpy as np

from vec_parallel_env.vector import stack, SubProcVecParallelEnv
from pettingzoo.test.example_envs import generated_agents_parallel_v0


def test_stack():
    a = {"agent_0": np.zeros((10)), "agent_1": np.zeros((10, 5))}
    b = {"agent_0": np.ones((10)), "agent_1": np.ones((10, 5))}
    c = stack([a, b])

    assert c["agent_0"].shape == (2, 10)
    assert c["agent_1"].shape == (2, 10, 5)


def generate_envs(n: int):
    env_fn = generated_agents_parallel_v0.parallel_env()
    envs = SubProcVecParallelEnv([lambda: env_fn for _ in range(5)])
    return envs


def test_vec_env_creation():
    envs = generate_envs(5)
    envs.close()
    assert True


def test_vec_env_step():
    N = 5
    envs = generate_envs(N)

    action_spaces = envs.action_spaces
    actions = {
        agent: np.stack([space.sample() for _ in range(N)])
        for agent, space in action_spaces.items()
    }
    obs, _, _, _, _ = envs.step(actions)

    assert len(obs[envs.agents[0]]) == N

    envs.close()
    assert True
