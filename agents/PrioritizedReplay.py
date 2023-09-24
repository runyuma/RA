from typing import Union
from typing import Any, Dict, Generator, List, Optional, Union
import io
import pathlib
from gymnasium import spaces
from stable_baselines3.common.buffers import ReplayBuffer, DictReplayBuffer
import torch as th
import numpy as np
from stable_baselines3.common.save_util import load_from_pkl, save_to_pkl

class PrioritizedReplayBuffer(DictReplayBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        super(PrioritizedReplayBuffer, self).__init__(
            buffer_size, observation_space, action_space, device, n_envs, optimize_memory_usage, handle_timeout_termination
        )
        self.priorities = np.zeros((buffer_size,), dtype=np.float32)
    
    def load_replay_buffer(
        self,
        path: Union[str, pathlib.Path, io.BufferedIOBase],
        truncate_last_traj: bool = True,
    ) -> None:
        """
        Load a replay buffer from a pickle file.

        :param path: Path to the pickled replay buffer.
        :param truncate_last_traj: When using ``HerReplayBuffer`` with online sampling:
            If set to ``True``, we assume that the last trajectory in the replay buffer was finished
            (and truncate it).
            If set to ``False``, we assume that we continue the same trajectory (same episode).
        """
        replay_buffer = load_from_pkl(path,1)
        length = replay_buffer.actions.shape[0]
        for key in self.observations.keys():
            self.observations[key][:length] = replay_buffer.observations[key]
            self.next_observations[key][:length] = replay_buffer.next_observations[key] 
        self.actions[:length] = replay_buffer.actions
        self.rewards[:length] = replay_buffer.rewards
        self.dones[:length] = replay_buffer.dones
        self.pos = length
    def sample(self, batch_size: int, env) :

        upper_bound = self.buffer_size if self.full else self.pos
        prob = np.exp(self.rewards[:upper_bound]).flatten()
        prob = prob / np.sum(prob)
        # print("prob shape",prob.shape)
        batch_inds = np.random.choice( upper_bound, size=batch_size,replace=False, p=prob)
        # print(batch_inds)
        # print("class",isinstance(self, DictReplayBuffer))
        # print("class",isinstance(ReplayBuffer, DictReplayBuffer))
        return self._get_samples(batch_inds, env=env)
