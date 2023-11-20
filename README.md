# vectorization_parallel_env
> Vectorize PettingZoo ParallelEnv

<details>
<summary> Installation</summary>

```bash
pip install git+https://github.com/Raffaelbdl/vectorization_parallel_env
```
</details>

## Example Usage
```python
from vec_parallel_env import SubProcVecParallelEnv
from pettingzoo.mpe import simple_spread_v3

env_fn = simple_spread_v3.parallel_env()
envs = SubProcVecParallelEnv([lambda: env_fn for _ in range(5)])

# training loop

envs.close()
```

## API details
- `envs.step ` takes a dictionary of batched actions
```python
action_spaces = envs.action_spaces
actions = {
    agent: np.stack([space.sample() for _ in range(envs.num_agents)])
    for agent, space in action_spaces.items()
}
``` 
- `envs.step ` returns a gymnasium-style tuple where each element is a dictionary of batched values
```python
observation_spaces = envs.observation_spaces
observations = {
    agent: np.stack([space.sample() for _ in range(envs.num_agents)])
    for agent, space in observation_spaces.items()
}
```
