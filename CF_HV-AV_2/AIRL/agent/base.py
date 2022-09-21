from abc import ABC, abstractmethod
import os
import numpy as np
import torch


class Algorithm(ABC):
    """
    Base class for all algorithms
    Parameters
    ----------
    state_shape: np.array
        shape of the state space
    action_shape: np.array
        shape of the action space
    device: torch.device
        cpu or cuda
    seed: int
        random seed
    gamma: float
        discount factor
    """
    def __init__(self, state_shape, action_shape, device, seed, gamma):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        self.learning_steps = 0
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.device = device
        self.gamma = gamma

    def explore(self, state):
        """
        Act with policy with randomness
        Parameters
        ----------
        state: np.array
            current state
        Returns
        -------
        action: np.array
            mean action
        log_pi: float
            log(\pi(a|s)) of the action
        """
        # next_states
        a = []
        for item in state:
            if isinstance(item, np.ndarray):
                a.append(item[0])
            else:
                a.append(item)
        state = np.array(a)

        state = torch.tensor(state, dtype=torch.float, device=self.device)
        with torch.no_grad():
            action, log_pi = self.actor.sample(state.unsqueeze_(0))
        return action.cpu().numpy()[0], log_pi.item()

    def exploit(self, state):
        """
        Act with deterministic policy
        Parameters
        ----------
        state: np.array
            current state
        Returns
        -------
        action: np.array
            action to take
        """
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        with torch.no_grad():
            action = self.actor(state.unsqueeze_(0))
        return action.cpu().numpy()[0]

    @abstractmethod
    def is_update(self, step):
        """
        Whether the time is for update
        Parameters
        ----------
        step: int
            current training step
        """
        pass

    @abstractmethod
    def update(self):
        """
        Update the algorithm
        Parameters
        ----------
        writer: SummaryWriter
            writer for logs
        """
        pass

    @abstractmethod
    def save_models(self, save_dir):
        """
        Save the model
        Parameters
        ----------
        save_dir: str
            path to save
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)


class Expert:
    """
    Base class for all well-trained algorithms
    Parameters
    ----------
    state_shape: np.array
        shape of the state space
    action_shape: np.array
        shape of the action space
    device: torch.device
        cpu or cuda
    """
    def __init__(self, state_shape: np.array, action_shape: np.array, device: torch.device):
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.device = device
        self.actor = None

    def exploit(self, state: np.array) -> np.array:
        """
        Act with deterministic policy
        Parameters
        ----------
        state: np.array
            current state
        Returns
        -------
        action: np.array
            action to take
        """
        state = torch.tensor(state, dtype=torch.float, device=self.device)

        with torch.no_grad():
            action = self.actor(state.unsqueeze_(0))

        return action.cpu().numpy()[0]
