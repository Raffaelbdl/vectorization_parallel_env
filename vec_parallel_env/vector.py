from abc import ABC, abstractmethod
from multiprocessing import Process, Pipe
from multiprocessing.connection import Connection
from typing import Any, Callable

import numpy as np
from pettingzoo import ParallelEnv


def tile_images(img_nhwc):
    """
    Tile N images into one big PxQ image
    (P,Q) are chosen to be as close as possible, and if N
    is square, then P=Q.

    input: img_nhwc, list or array of images, ndim=4 once turned into array
        n = batch index, h = height, w = width, c = channel
    returns:
        bigim_HWc, ndarray with ndim=3
    """
    img_nhwc = np.asarray(img_nhwc)
    N, h, w, c = img_nhwc.shape
    H = int(np.ceil(np.sqrt(N)))
    W = int(np.ceil(float(N) / H))
    img_nhwc = np.array(list(img_nhwc) + [img_nhwc[0] * 0 for _ in range(N, H * W)])
    img_HWhwc = img_nhwc.reshape(H, W, h, w, c)
    img_HhWwc = img_HWhwc.transpose(0, 2, 1, 3, 4)
    img_Hh_Ww_c = img_HhWwc.reshape(H * h, W * w, c)
    return img_Hh_Ww_c


class CloudPickleWrapper:
    """Uses cloudpickle to serialize contents"""

    def __init__(self, x) -> None:
        self.x = x

    def __getstate__(self):
        import cloudpickle

        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle

        self.x = pickle.loads(ob)


class VecParallelEnv(ABC):
    """An abstract asynchronous, vectorized environment"""

    def __init__(self, num_envs: int, observation_spaces, action_spaces) -> None:
        self.num_envs = num_envs
        self.observation_spaces = observation_spaces
        self.action_spaces = action_spaces

    @abstractmethod
    def reset(self):
        """
        Reset all the environments and return an array of
        observations, or a dict of observation arrays.

        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.
        """
        pass

    @abstractmethod
    def step_async(self, actions):
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.

        You should not call this if a step_async run is
        already pending.
        """
        pass

    @abstractmethod
    def step_wait(self):
        """
        Wait for the step taken with step_async().

        Returns (obs, rews, dones, infos):
         - obs: an array of observations, or a dict of
                arrays of observations.
         - rews: an array of rewards
         - dones: an array of "episode done" booleans
         - infos: a sequence of info objects
        """
        pass

    def close_extras(self):
        """
        Clean up the  extra resources, beyond what's in this base class.
        Only runs when not self.closed.
        """
        pass

    def close(self):
        if self.closed:
            return
        if self.viewer is not None:
            self.viewer.close()
        self.close_extras()
        self.closed = True

    def step(self, actions):
        """
        Step the environments synchronously.

        This is available for backwards compatibility.
        """
        self.step_async(actions)
        return self.step_wait()

    def render(self, mode="human"):
        imgs = self.get_images()
        bigimg = tile_images(imgs)
        if mode == "human":
            self.get_viewer().imshow(bigimg)
            return self.get_viewer().isopen
        elif mode == "rgb_array":
            return bigimg
        else:
            raise NotImplementedError

    def get_images(self):
        """
        Return RGB images from each environment
        """
        raise NotImplementedError


def worker(child_conn: Connection, parent_conn: Connection, env_fn_wrapper):
    parent_conn.close()
    env: ParallelEnv = env_fn_wrapper.x()  # using cloudpickle

    while True:
        cmd, data = child_conn.recv()

        if cmd == "close":
            env.close()
            child_conn.close()
            break

        elif cmd == "get_agents":
            child_conn.send(env.agents)
        elif cmd == "get_num_agents":
            child_conn.send((env.num_agents))
        elif cmd == "get_spaces":
            observation_spaces = {
                agent: env.observation_space(agent) for agent in env.agents
            }
            action_spaces = {agent: env.action_space(agent) for agent in env.agents}
            child_conn.send((observation_spaces, action_spaces))

        elif cmd == "step":
            s, r, d, t, i = env.step(data)
            if any(d.values()) or any(t.values()):
                s, i = env.reset()
            child_conn.send((s, r, d, t, i))
        elif cmd == "reset":
            child_conn.send(env.reset())

        else:
            raise NotImplementedError


class SubProcVecParallelEnv(VecParallelEnv):
    """
    VecEnv that runs multiple environments in parallel in subproceses and communicates with them via pipes.
    Recommended to use when num_envs > 1 and step() can be a bottleneck.
    """

    def __init__(self, env_fns: Callable[[Any], ParallelEnv]) -> None:
        self.waiting = False
        self.closed = False

        num_envs = len(env_fns)
        self.parent_conns, self.child_conns = zip(*[Pipe() for _ in range(num_envs)])
        self.ps = [
            Process(
                target=worker,
                args=(child_conn, parent_conn, CloudPickleWrapper(env_fn)),
            )
            for (parent_conn, child_conn, env_fn) in zip(
                self.parent_conns, self.child_conns, env_fns
            )
        ]
        for p in self.ps:
            # if the main process crashes, we should not cause things to hang
            p.daemon = True
            p.start()

        for conn in self.child_conns:
            conn.close()

        for conn in self.parent_conns:
            conn.send(("reset", None))
            conn.recv()

        self.parent_conns[0].send((("get_agents", None)))
        self.agents = self.parent_conns[0].recv()

        self.parent_conns[0].send((("get_num_agents", None)))
        self.num_agents = self.parent_conns[0].recv()

        self.parent_conns[0].send(("get_spaces", None))
        observation_spaces, action_spaces = self.parent_conns[0].recv()

        VecParallelEnv.__init__(self, num_envs, observation_spaces, action_spaces)

    def step_async(self, actions):
        unstack_actions = []
        for i in range(self.num_agents):
            unstack_actions.append(
                {agent: action[i] for agent, action in actions.items()}
            )
        for conn, action in zip(self.parent_conns, unstack_actions):
            conn.send(("step", action))
        self.waiting = True

    def step_wait(self):
        results = [conn.recv() for conn in self.parent_conns]
        self.waiting = False
        s, r, d, t, i = zip(*results)
        return stack(s), stack(r), stack(d), stack(t), i

    def reset(self):
        for conn in self.parent_conns:
            conn.send(("reset", None))
        results = [conn.recv() for conn in self.parent_conns]
        s, i = zip(*results)
        return stack(s), i

    def close(self):
        if self.closed:
            return

        if self.waiting:
            for conn in self.parent_conns:
                conn.recv()

        for conn in self.parent_conns:
            conn.send(("close", None))

        for p in self.ps:
            p.join()

        self.closed = True


def stack(xs: list[dict[str, np.ndarray]]):
    ks = xs[0].keys()
    vs = list(zip(*[x.values() for x in xs]))
    return {k: np.stack(vs[i]) for i, k in enumerate(ks)}
