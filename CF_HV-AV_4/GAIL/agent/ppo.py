import os
import torch
from torch import nn
from torch.optim import Adam
import numpy as np
from .base import Algorithm, Expert
from buffer import RolloutBuffer
from network import StateIndependentPolicy, StateFunction
from utils import disable_gradient


def calculate_gae(values, rewards, dones, next_values, gamma, lambd):
    # Calculate TD errors.
    deltas = rewards + gamma * next_values * (1 - dones) - values
    # Initialize gae.
    gaes = torch.empty_like(rewards)

    # Calculate gae recursively from behind.
    gaes[-1] = deltas[-1]
    for t in reversed(range(rewards.size(0) - 1)):
        gaes[t] = deltas[t] + gamma * lambd * (1 - dones[t]) * gaes[t + 1]

    return gaes + values, (gaes - gaes.mean()) / (gaes.std() + 1e-8)


class PPO(Algorithm):
    """
    Implementation of PPO
    Reference:
    ----------
    [1] Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and Klimov, O.
    Proximal policy optimization algorithms.
    arXiv preprint arXiv:1707.06347, 2017.
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
    rollout_length: int
        rollout length of the buffer
    mix_buffer: int
        times for rollout buffer to mix
    lr_actor: float
        learning rate of the actor
    lr_critic: float
        learning rate of the critic
    units_actor: tuple
        hidden units of the actor
    units_critic: tuple
        hidden units of the critic
    epoch_ppo: int
        at each update period, update ppo for these times
    clip_eps: float
        clip coefficient in PPO's objective
    lambd: float
        lambd factor
    coef_ent: float
        entropy coefficient
    max_grad_norm: float
        maximum gradient norm
    """
    def __init__(self, state_shape, action_shape, device, seed, gamma=0.995,
                 rollout_length=200, mix_buffer=1, lr_actor=3e-4,
                 lr_critic=3e-4, units_actor=(64, 64), units_critic=(64, 64),
                 epoch_ppo=10, clip_eps=0.1, lambd=0.97, coef_ent=0,
                 max_grad_norm=10.0):
        super().__init__(state_shape, action_shape, device, seed, gamma)

        # Rollout buffer.
        self.buffer = RolloutBuffer(
            buffer_size=rollout_length,
            state_shape=state_shape,
            action_shape=action_shape,
            device=device,
            mix=mix_buffer
        )

        # Actor.
        self.actor = StateIndependentPolicy(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=units_actor,
            hidden_activation=nn.Tanh()
        ).to(device)

        # Critic.
        self.critic = StateFunction(
            state_shape=state_shape,
            hidden_units=units_critic,
            hidden_activation=nn.Tanh()
        ).to(device)

        self.optim_actor = Adam(self.actor.parameters(), lr=lr_actor)
        self.optim_critic = Adam(self.critic.parameters(), lr=lr_critic)

        self.learning_steps_ppo = 0
        self.rollout_length = rollout_length
        self.epoch_ppo = epoch_ppo
        self.clip_eps = clip_eps
        self.lambd = lambd
        self.coef_ent = coef_ent
        self.max_grad_norm = max_grad_norm

    def is_update(self, step):
        """
        Whether the time is for update
        Parameters
        ----------
        step: int
            current training step
        Returns
        -------
        update: bool
            whether to update. PPO updates when the rollout buffer is full.
        """
        return step % self.rollout_length == 0 and step >= self.rollout_length

    def step(self, env, state, t):
        """
        Sample one step in the environment
        Parameters
        ----------
        env: gym.wrappers.TimeLimit
            environment
        state: np.array
            current state
        t: int
            current time step in the episode
        Returns
        -------
        next_state: np.array
            next state
        done: bool
            finished or not
        t: int
            time step
        """
        t += 1

        # state
        a = []
        for item in state:
            if isinstance(item, np.ndarray):
                a.append(item[0])
            else:
                a.append(item)
        state = np.array(a)
        action, log_pi = self.explore(state)
        next_state, reward, done, _ = env.step(action)
        self.buffer.append(state, action, reward, done, log_pi, next_state)

        return next_state, done, t

    def update(self, writer):
        """
        Update the algorithm
        Parameters
        ----------
        writer: SummaryWriter
            writer for logs
        """
        self.learning_steps += 1
        states, actions, rewards, dones, log_pis, next_states = \
            self.buffer.get()
        self.update_ppo(
            states, actions, rewards, dones, log_pis, next_states, writer)

    def update_ppo(self, states, actions, rewards, dones, log_pis, next_states,
                   writer):
        """
        Update PPO's actor and critic for some steps
        Parameters
        ----------
        states: torch.Tensor
            sampled states
        actions: torch.Tensor
            sampled actions according to the states
        rewards: torch.Tensor
            rewards of the s-a pairs
        dones: torch.Tensor
            whether is the end of the episode
        log_pis: torch.Tensor
            log(\pi(a|s)) of the actions
        next_states: torch.Tensor
            next states give s-a pairs
        writer: SummaryWriter
            writer for logs
        """
        with torch.no_grad():
            values = self.critic(states)
            next_values = self.critic(next_states)

        targets, gaes = calculate_gae(
            values, rewards, dones, next_values, self.gamma, self.lambd)

        for _ in range(self.epoch_ppo):
            self.learning_steps_ppo += 1
            self.update_critic(states, targets, writer)
            self.update_actor(states, actions, log_pis, gaes, writer)

    def update_critic(self, states, targets, writer):
        """
        Update the critic for one step
        Parameters
        ----------
        states: torch.Tensor
            sampled states
        targets: torch.Tensor
            advantages
        writer: SummaryWriter
            writer for logs
        """
        loss_critic = (self.critic(states) - targets).pow_(2).mean()

        self.optim_critic.zero_grad()
        loss_critic.backward(retain_graph=False)
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.optim_critic.step()

        if self.learning_steps_ppo % self.epoch_ppo == 0:
            writer.add_scalar(
                'loss/critic', loss_critic.item(), self.learning_steps)

    def update_actor(self, states, actions, log_pis_old, gaes, writer):
        """
        Update the actor for one step
        Parameters
        ----------
        states: torch.Tensor
            sampled states
        actions: torch.Tensor
            sampled actions according to the states
        log_pis_old: torch.Tensor
            log(\pi(a|s)) of the previous action
        gaes: torch.Tensor
            advantages
        writer: SummaryWriter
            writer for logs
        """
        log_pis = self.actor.evaluate_log_pi(states, actions)
        entropy = -log_pis.mean()

        ratios = (log_pis - log_pis_old).exp_()
        loss_actor1 = -ratios * gaes
        loss_actor2 = -torch.clamp(
            ratios,
            1.0 - self.clip_eps,
            1.0 + self.clip_eps
        ) * gaes
        loss_actor = torch.max(loss_actor1, loss_actor2).mean()

        self.optim_actor.zero_grad()
        (loss_actor - self.coef_ent * entropy).backward(retain_graph=False)
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.optim_actor.step()

        if self.learning_steps_ppo % self.epoch_ppo == 0:
            writer.add_scalar(
                'loss/actor', loss_actor.item(), self.learning_steps)
            writer.add_scalar(
                'stats/entropy', entropy.item(), self.learning_steps)

    def save_models(self, save_dir: str):
        """
        Save the model
        Parameters
        ----------
        save_dir: str
            path to save
        """
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        torch.save(self.actor.state_dict(), f'{save_dir}/actor.pkl')


class PPOExpert(Expert):
    """
    Well-trained PPO agent
    Parameters
    ----------
    state_shape: np.array
        shape of the state space
    action_shape: np.array
        shape of the action space
    device: torch.device
        cpu or cuda
    path: str
        path to the well-trained weights
    units_actor: tuple
        hidden units of the actor
    """
    def __init__(
            self,
            state_shape: np.array,
            action_shape: np.array,
            device: torch.device,
            path: str,
            units_actor: tuple = (64, 64)
    ):
        super(PPOExpert, self).__init__(
            state_shape=state_shape,
            action_shape=action_shape,
            device=device
        )

        self.actor = StateIndependentPolicy(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=units_actor,
            hidden_activation=nn.Tanh()
        ).to(device)
        self.actor.load_state_dict(torch.load(path, map_location=device))
        disable_gradient(self.actor)