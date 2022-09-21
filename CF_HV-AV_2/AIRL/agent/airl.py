import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
from .ppo import PPO, PPOExpert
from network import AIRLDiscrim
import os


class AIRL(PPO):
    """
    Implementation of AIRL, using PPO as the backbone RL algorithm
    Reference:
    ----------
    [1] Fu, J., Luo, K., and Levine, S.
    Learning robust rewards with adversarial inverse reinforcement learning.
    In International Conference on Learning Representations, 2018.
    Parameters
    ----------
    buffer_exp: SerializedBuffer
        buffer of demonstrations
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
    batch_size: int
        batch size for sampling from current policy and demonstrations
    lr_actor: float
        learning rate of the actor
    lr_critic: float
        learning rate of the critic
    lr_disc: float
        learning rate of the discriminator
    units_actor: tuple
        hidden units of the actor
    units_critic: tuple
        hidden units of the critic
    units_disc_r: tuple
        hidden units of the discriminator r
    units_disc_v: tuple
        hidden units of the discriminator v
    epoch_ppo: int
        at each update period, update ppo for these times
    epoch_disc: int
        at each update period, update the discriminator for these times
    clip_eps: float
        clip coefficient in PPO's objective
    lambd: float
        lambd factor
    coef_ent: float
        entropy coefficient
    max_grad_norm: float
        maximum gradient norm
    """
    def __init__(self, buffer_exp, state_shape, action_shape, device, seed,
                 gamma=0.995, rollout_length=200, mix_buffer=1,
                 batch_size=64, lr_actor=1e-4, lr_critic=1e-4, lr_disc=1e-4,
                 units_actor=(32, 32), units_critic=(32, 32),
                 units_disc_r=(64, ), units_disc_v=(64, ),
                 epoch_ppo=6, epoch_disc=6, clip_eps=0.4, lambd=0.97,
                 coef_ent=0.03, max_grad_norm=10.0):
        super().__init__(
            state_shape, action_shape, device, seed, gamma, rollout_length,
            mix_buffer, lr_actor, lr_critic, units_actor, units_critic,
            epoch_ppo, clip_eps, lambd, coef_ent, max_grad_norm
        )

        # Expert's buffer.
        self.buffer_exp = buffer_exp

        # Discriminator.
        self.disc = AIRLDiscrim(
            state_shape=state_shape,
            gamma=gamma,
            hidden_units_r=units_disc_r,
            hidden_units_v=units_disc_v,
            hidden_activation_r=nn.ReLU(inplace=True),
            hidden_activation_v=nn.ReLU(inplace=True)
        ).to(device)

        self.learning_steps_disc = 0
        self.optim_disc = Adam(self.disc.parameters(), lr=lr_disc)
        self.batch_size = batch_size
        self.epoch_disc = epoch_disc

    def update(self, writer):
        """
        Update the algorithm
        Parameters
        ----------
        writer: SummaryWriter
            writer for logs
        """
        self.learning_steps += 1

        for _ in range(self.epoch_disc):
            self.learning_steps_disc += 1

            # Samples from current policy's trajectories.
            states, _, _, dones, log_pis, next_states = self.buffer.sample(self.batch_size)
            # Samples from expert's demonstrations.
            states_exp, actions_exp, _, dones_exp, next_states_exp = \
                self.buffer_exp.sample(self.batch_size)
            # Calculate log probabilities of expert actions.
            with torch.no_grad():
                log_pis_exp = self.actor.evaluate_log_pi(
                    states_exp, actions_exp)
            # Update discriminator.
            self.update_disc(
                states, dones, log_pis, next_states, states_exp,
                dones_exp, log_pis_exp, next_states_exp, writer
            )

        # We don't use reward signals here,
        states, actions, _, dones, log_pis, next_states = self.buffer.get()

        # Calculate rewards.
        rewards = self.disc.calculate_reward(
            states, dones, log_pis, next_states)

        # Update PPO using estimated rewards.
        self.update_ppo(
            states, actions, rewards, dones, log_pis, next_states, writer)

    def update_disc(self, states, dones, log_pis, next_states,
                    states_exp, dones_exp, log_pis_exp,
                    next_states_exp, writer):
        """
        Train the discriminator to distinguish the expert's behavior
        and the imitation learning policy's behavior
        Parameters
        ----------
        states: torch.Tensor
            states sampled from current IL policy
        dones: torch.Tensor
            dones sampled from current IL policy
        log_pis: torch.Tensor
            log(\pi(s|a)) sampled from current IL policy
        next_states: torch.Tensor
            next states sampled from current IL policy
        states_exp: torch.Tensor
            states sampled from demonstrations
        dones_exp: torch.Tensor
            dones sampled from demonstrations
        log_pis_exp: torch.Tensor
            log(\pi(s|a)) sampled from demonstrations
        next_states_exp: torch.Tensor
            next states sampled from demonstrations
        writer: SummaryWriter
            writer for logs
        """
        # Output of discriminator is (-inf, inf), not [0, 1].
        logits_pi = self.disc(states, dones.squeeze(1), log_pis, next_states)
        logits_exp = self.disc(
            states_exp, dones_exp, log_pis_exp, next_states_exp)

        # Discriminator is to maximize E_{\pi} [log(1 - D)] + E_{exp} [log(D)].
        loss_pi = -F.logsigmoid(-logits_pi).mean()
        loss_exp = -F.logsigmoid(logits_exp).mean()
        loss_disc = loss_pi + loss_exp

        self.optim_disc.zero_grad()
        loss_disc.backward()
        self.optim_disc.step()

        if self.learning_steps_disc % self.epoch_disc == 0:
            writer.add_scalar('loss/pi', loss_pi.item(), self.learning_steps)
            writer.add_scalar('loss/exp', loss_exp.item(), self.learning_steps)
            writer.add_scalar('loss/disc', loss_disc.item(), self.learning_steps)
            print('AIRL\tloss_pi: {:.6f}\tloss_exp: {:.6f}\tloss_disc: {:.6f}\t'.format(loss_pi, loss_exp, loss_disc))

            # Discriminator's accuracies.
            with torch.no_grad():
                acc_pi = (logits_pi < 0).float().mean().item()
                acc_exp = (logits_exp > 0).float().mean().item()
            writer.add_scalar('stats/acc_pi', acc_pi, self.learning_steps)
            writer.add_scalar('stats/acc_exp', acc_exp, self.learning_steps)
            # print('AIRL\tstats/acc_pi: {:.6f}\tstats/acc_exp: {:.6f}\t'.format(acc_pi, acc_exp))

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
        torch.save(self.disc.state_dict(), f'{save_dir}/airl_disc.pkl')
        torch.save(self.actor.state_dict(), f'{save_dir}/airl_actor.pkl')


class AIRLExpert(PPOExpert):
    """
    Well-trained AIRL agent
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
            units_actor: tuple = (32, 32)
    ):
        super(AIRLExpert, self).__init__(
            state_shape=state_shape,
            action_shape=action_shape,
            device=device,
            path=path+'airl_actor.pkl',
            units_actor=units_actor
        )