import codecs
import contextlib
import json

from typing import Protocol

import numpy as np


class ReplayBuffer:
    """A flexible replay buffer"""

    def __init__(self, schema, capacity=0):
        self.schema = {
            k: (tuple(shape), str(dtype)) for k, (shape, dtype) in dict(schema).items()
        }
        self.schema["_id"] = ((), "uint64")
        self.schema["_step"] = ((), "uint16")

        self.columns = tuple(self.schema)

        self._size = 0
        self._capacity = 0
        self._next_id = 0
        self._last_id = None
        self._data = {k: [] for k in self.schema}

        self._episode = None

    @classmethod
    def load(cls, path):
        try:
            import tables
            from tables.nodes import filenode

        except ImportError as err:
            raise RuntimeError("IO requires pytables (pip install tables)") from err

        with tables.open_file(path, "r") as h5:
            with filenode.open_node(h5.get_node("/", name="schema")) as fobj:
                fobj = codecs.getreader("utf8")(fobj)
                schema = json.load(fobj)

            data = {}

            for k in schema:
                n = h5.get_node("/data", k)
                data[k] = np.asarray(n)
                n.close()

        buffer = cls(schema)
        buffer._data = data
        buffer._size = len(data["_id"])
        buffer._capacity = len(data["_id"])
        buffer._next_id = data["_id"].max() + 1

        return buffer

    def save(self, path):
        try:
            import tables
            from tables.nodes import filenode

        except ImportError as err:
            raise RuntimeError("IO requires pytables (pip install tables)") from err

        with tables.open_file(path, "w") as h5:
            with filenode.new_node(h5, where="/", name="schema") as fobj:
                fobj = codecs.getwriter("utf8")(fobj)
                json.dump(self.schema, fobj, indent=2)

            h5.create_group("/", "data")

            for k, v in self._data.items():
                h5.create_array("/data", k, v[: self._size])

    @property
    def c(self):
        return ReplayBufferColumns(self)

    def add(self, **values):
        assert self._episode is not None
        self._episode.append(values)

    def add_episode(self, episode, finalize=None):
        self.begin()
        try:
            self._episode = list(episode)
            self.commit(finalize=finalize)

        except Exception:
            self.rollback()
            raise

    def begin(self):
        assert self._episode is None
        self._episode = []

    def commit(self, *, finalize=None):
        num_steps, data = self._build_data(finalize)

        self.reserve(self._size + num_steps)
        for k, v in data.items():
            self._data[k][self._size : self._size + len(v)] = v

        # perform any permant changes last
        self._episode = None
        self._size += num_steps
        self._last_id = self._next_id
        self._next_id += 1

    def _build_data(self, finalize=None):
        data = {}
        num_steps = len(self._episode)
        for step in self._episode:
            for k, v in step.items():
                data.setdefault(k, []).append(v)

        assert len({len(v) for v in data.values()}) == 1

        if finalize is not None:
            data = finalize(data)

        data["_id"] = [self._next_id for _ in range(num_steps)]
        data["_step"] = np.arange(num_steps)

        assert set(data) == set(self.schema)

        return num_steps, data

    def reserve(self, min_capacity):
        min_capacity = max(1024, min_capacity)

        if self._capacity >= min_capacity:
            return

        new_capacity = max(min_capacity, int(1.2 * self._capacity))

        try:
            for k, (shape, dtype) in self.schema.items():
                old_data = self._data.get(k, [])
                new_data = np.empty((new_capacity,) + shape, dtype=dtype)
                new_data[: self._capacity] = old_data

                self._data[k] = new_data

        except Exception:
            self._data = {k: v[: self._capacity] for k, v in self._data.items()}
            raise

        self._capacity = new_capacity

    def rollback(self):
        self._episode = None

    def get(self, *keys):
        res = []
        for k in keys:
            res.append(self._data[k][: self._size])

        return tuple(res)

    def __repr__(self):
        return "<ReplayBuffer {}>".format(", ".join(self.schema))

    def __len__(self):
        return self._size

    def __getitem__(self, idx):
        assert idx < self._size
        return {k: v[idx] for k, v in self._data.items()}

    def episode(self, func=None, *, finalize=None):
        if func is None:
            return self._episode_ctx(finalize=finalize)

        with self._episode_ctx(finalize=finalize):
            func()

    @contextlib.contextmanager
    def _episode_ctx(self, *, finalize=None):
        self.begin()
        try:
            yield
            self.commit(finalize=finalize)

        except:
            self.rollback()
            raise

    def to_dataset(self, keys=("obs", "next_obs", "action", "reward", "done")):
        from chmp.torch_utils import NumpyDataset

        data = {key: getattr(self.c, key) for key in keys}
        return NumpyDataset(data)


class ReplayBufferColumns:
    def __init__(self, buffer):
        self._buffer = buffer

    def __getattr__(self, attr):
        res = self._buffer.get(attr)
        return res[0]

    def __len__(self):
        return len(self._buffer.columns)

    def __getitem__(self, i):
        return self._buffer.columns[i]

    def __repr__(self):
        return "<ReplayBufferColumn {}>".format(", ".join(self._buffer.columns))

    def __dir__(self):
        return tuple(self._buffer.columns)


class FunctionalEnv:
    """A environment that is implement in terms of pure functions

    In contrast to standard OpenAI gym envs the state is explicitly passed from
    function to function.
    """

    def __init__(self):
        self._impl = self.get_impl()
        self._state = None

    def get_impl(self) -> "FunctionalEnvImpl":
        raise NotImplementedError()

    def reset(self):
        self._state = self._impl.init()
        obs = self._impl.observe(self._state)
        return obs

    def step(self, action):
        done, next_state = self._impl.transition(self._state, action)
        reward = self._impl.reward(self._state, next_state, action)
        obs = self._impl.observe(next_state)

        self._state = next_state

        return obs, reward, done, None


class FunctionalEnvImpl(Protocol):
    def init(self):
        ...

    def observe(self, state):
        ...

    def reward(self, prev_state, state, action):
        ...

    def transition(self, state, action):
        ...

    def sample_action(self, state):
        ...


def add_reward_to_go(
    episode, reward_column="reward", reward_to_go_column="reward_to_go"
):
    """Return a new dict with the reward_to_go added as a new column."""
    episode = dict(episode)
    episode[reward_to_go_column] = np.cumsum(episode[reward_column][::-1])[::-1]
    return episode


def simulate(env, agent, max_steps=1024):
    episode = []
    obs = env.reset()

    for _ in range(max_steps):
        action = agent.act(obs)
        new_obs, reward, done, _ = env.step(action)

        step = dict(
            obs=obs,
            next_obs=new_obs,
            action=action,
            reward=reward,
            done=done,
        )
        episode.append(step)
        obs = new_obs

        if done:
            break

    return episode
