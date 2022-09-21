import os
import numpy as np
import torch
from typing import Tuple

class SerializedBuffer:
    """
    Serialized buffer, containing [states, actions, rewards, done, next_states]
     and trajectories, often used as demonstrations
    Parameters
    ----------
    path: str
        path to the saved buffer
    device: torch.device
        cpu or cuda
    """
    def __init__(
            self,
            path: str,
            device: torch.device,):
        tmp = np.load(path, allow_pickle=True)
        self.buffer_size = self._n = len(tmp.item()['states'])
        self.device = device

        # batch_state
        a = []
        b = []
        for items in tmp.item()['states']:
            for item in items:
                for num in item:
                    if isinstance(num, float):
                        a.append(num)
                    else:
                        a.append(item)
                b.append(np.array(a)[:3])
                a = []
        batch_state = np.array(b)

        # actions
        a = []
        b = []
        for items in tmp.item()['actions']:
            for item in items:
                if isinstance(item, str):
                    a.append(np.float(item))
                else:
                    a.append(item)
            b.append(a)
            a = []
        batch_action = np.array(b)

        # batch_next_state
        a = []
        b = []
        for items in tmp.item()['next_states']:
            for item in items:
                for num in item:
                    if isinstance(num, float):
                        a.append(num)
                    else:
                        a.append(item)
                b.append(np.array(a)[:3])
                a = []
        batch_next_state = np.array(b)

        self.states = torch.Tensor(batch_state).clone().to(self.device)
        self.actions = torch.cat([torch.as_tensor(i) for i in batch_action]).reshape(-1, 1).clone().to(self.device)
        self.rewards = torch.cat([torch.as_tensor(i) for i in tmp.item()['rewards']]).clone().to(self.device)
        self.dones = torch.cat([torch.as_tensor(i) for i in tmp.item()['dones']]).clone().to(self.device)
        self.next_states = torch.Tensor(batch_next_state).clone().to(self.device)

    def sample(
            self,
            batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample data from the buffer
        Parameters
        ----------
        batch_size: int
            batch size
        Returns
        -------
        states: torch.Tensor
        actions: torch.Tensor
        rewards: torch.Tensor
        dones: torch.Tensor
        next_states: torch.Tensor
        """
        idxes = np.random.randint(low=0, high=self._n, size=batch_size)
        return (
            self.states[idxes],
            self.actions[idxes],
            self.rewards[idxes],
            self.dones[idxes],
            self.next_states[idxes]
        )


class Buffer(SerializedBuffer):
    """
    Buffer used while collecting demonstrations
    Parameters
    ----------
    buffer_size: int
        size of the buffer
    state_shape: np.array
        shape of the state space
    action_shape: np.array
        shape of the action space
    device: torch.device
        cpu or cuda
    """
    def __init__(
            self,
            buffer_size: int,
            state_shape: np.array,
            action_shape: np.array,
            device: torch.device
    ):
        self._n = 0
        self._p = 0
        self.buffer_size = buffer_size
        self.device = device

        self.states = torch.empty(
            (buffer_size, *state_shape), dtype=torch.float, device=device)
        self.actions = torch.empty(
            (buffer_size, *action_shape), dtype=torch.float, device=device)
        self.rewards = torch.empty(
            (buffer_size, 1), dtype=torch.float, device=device)
        self.dones = torch.empty(
            (buffer_size, 1), dtype=torch.float, device=device)
        self.next_states = torch.empty(
            (buffer_size, *state_shape), dtype=torch.float, device=device)

    def append(
            self,
            state: np.array,
            action: np.array,
            reward: float,
            done: bool,
            next_state: np.array):
        """
        Save a transition in the buffer
        Parameters
        ----------
        state: np.array
            current state
        action: np.array
            action taken in the state
        reward: float
            reward of the s-a pair
        done: bool
            whether the state is the end of the episode
        next_state: np.array
            next states that the s-a pair transferred to
        """
        self.states[self._p].copy_(torch.from_numpy(state))
        self.actions[self._p].copy_(torch.from_numpy(action))
        self.rewards[self._p] = float(reward)
        self.dones[self._p] = float(done)
        self.next_states[self._p].copy_(torch.from_numpy(next_state))

        self._p = (self._p + 1) % self.buffer_size
        self._n = min(self._n + 1, self.buffer_size)

    def save(self, path):
        """
        Save the buffer
        Parameters
        ----------
        path: str
            path to save
        """
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        torch.save({
            'state': self.states.clone().cpu(),
            'action': self.actions.clone().cpu(),
            'reward': self.rewards.clone().cpu(),
            'done': self.dones.clone().cpu(),
            'next_state': self.next_states.clone().cpu(),
        }, path)


class RolloutBuffer:
    """
    Rollout buffer that often used in training RL agents
    Parameters
    ----------
    buffer_size: int
        size of the buffer
    state_shape: np.array
        shape of the state space
    action_shape: np.array
        shape of the action space
    device: torch.device
        cpu or cuda
    mix: int
        the buffer will be mixed using these time of data
    """
    def __init__(self, buffer_size, state_shape, action_shape, device, mix=1):
        self._n = 0
        self._p = 0
        self.mix = mix
        self.buffer_size = buffer_size
        self.total_size = mix * buffer_size

        self.states = torch.empty(
            (self.total_size, *state_shape), dtype=torch.float, device=device)
        self.actions = torch.empty(
            (self.total_size, *action_shape), dtype=torch.float, device=device)
        self.rewards = torch.empty(
            (self.total_size, 1), dtype=torch.float, device=device)
        self.dones = torch.empty(
            (self.total_size, 1), dtype=torch.float, device=device)
        self.log_pis = torch.empty(
            (self.total_size, 1), dtype=torch.float, device=device)
        self.next_states = torch.empty(
            (self.total_size, *state_shape), dtype=torch.float, device=device)

    def append(self, state, action, reward, done, log_pi, next_state):
        """
        Save a transition in the buffer
        Parameters
        ----------
        state: np.array
            current state
        action: np.array
            action taken in the state
        reward: float
            reward of the s-a pair
        done: bool
            whether the state is the end of the episode
        log_pi: float
            log(\pi(a|s))
        next_state: np.array
            next states that the s-a pair transferred to
        """
        # next_states
        a = []
        for item in state:
            if isinstance(item, np.ndarray):
                a.append(item[0])
            else:
                a.append(item)
        state = np.array(a)

        self.states[self._p].copy_(torch.from_numpy(state))
        self.actions[self._p].copy_(torch.from_numpy(action))
        self.rewards[self._p] = float(reward)
        self.dones[self._p] = float(done)
        self.log_pis[self._p] = float(log_pi)

        # next_states
        a = []
        for item in next_state:
            if isinstance(item, np.ndarray):
                a.append(item[0])
            else:
                a.append(item)
        next_state = np.array(a)
        self.next_states[self._p].copy_(torch.from_numpy(next_state))

        self._p = (self._p + 1) % self.total_size
        self._n = min(self._n + 1, self.total_size)

    def get(self):
        """
        Get all data in the buffer
        Returns
        -------
        states: torch.Tensor
        actions: torch.Tensor
        rewards: torch.Tensor
        dones: torch.Tensor
        log_pis: torch.Tensor
        next_states: torch.Tensor
        """
        assert self._p % self.buffer_size == 0
        start = (self._p - self.buffer_size) % self.total_size
        idxes = slice(start, start + self.buffer_size)
        return (
            self.states[idxes],
            self.actions[idxes],
            self.rewards[idxes],
            self.dones[idxes],
            self.log_pis[idxes],
            self.next_states[idxes]
        )

    def sample(self, batch_size):
        """
        Sample data from the buffer
        Parameters
        ----------
        batch_size: int
            batch size
        Returns
        -------
        states: torch.Tensor
        actions: torch.Tensor
        rewards: torch.Tensor
        dones: torch.Tensor
        log_pis: torch.Tensor
        next_states: torch.Tensor
        """
        assert self._p % self.buffer_size == 0
        idxes = np.random.randint(low=0, high=self._n, size=batch_size)
        return (
            self.states[idxes],
            self.actions[idxes],
            self.rewards[idxes],
            self.dones[idxes],
            self.log_pis[idxes],
            self.next_states[idxes]
        )
