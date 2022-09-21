import os
import sys
from pathlib import Path
import random
import types

import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter

from logger import Logger
sys.path.append(str(Path(os.getcwd()).parent.parent.absolute()))
from CF_Env import CF_Env
from memory import Memory
from agent import sac
from utils import eval_mode, get_concat_samples, soft_update, hard_update
import matplotlib.pyplot as plt

torch.set_num_threads(2)

def get_args(cfg: DictConfig):
    cfg.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    os.chdir(str(Path(os.getcwd()).parent.parent.absolute()))
    cfg.hydra_base_dir = os.getcwd()
    # print(OmegaConf.to_yaml(cfg))
    return cfg

@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    args = get_args(cfg)

    # set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # load the environment
    env_args = args.env
    env = CF_Env()

    REPLAY_MEMORY = int(env_args.replay_mem)
    INITIAL_MEMORY = int(env_args.initial_mem)
    EPISODE_STEPS = int(env_args.eps_steps)
    LEARN_EPISODES = int(env_args.learn_episodes)

    # load the agent
    agent = sac.SAC(env.observation_space.shape[0], env.action_space.shape[0],
                    [env.action_LowerBound, env.action_HigherBound],
                    args.train.batch, args)

    # prepare the expert buffer
    expert_memory_replay = Memory(REPLAY_MEMORY//2, args.seed)
    expert_memory_replay.load(str(Path(os.getcwd()).parent.parent.absolute())+'/train_cf_3.npy',
                              num_trajs=args.eval.demos,
                              sample_freq=args.eval.subsample_freq,
                              seed=args.seed + 42)
    print(f'--> Expert memory size: {expert_memory_replay.size()}')

    # prepare the policy buffer
    online_memory_replay = Memory(REPLAY_MEMORY//2, args.seed+1)

    # prepare the save path
    log_dir = os.path.join(args.log_dir)
    writer = SummaryWriter(log_dir=log_dir)
    print(f'--> Saving logs at: {log_dir}')
    logger = Logger(args.log_dir)

    global train_set, trainNum, test_set, testNum
    # load training dataset
    train_set = np.load(str(Path(os.getcwd()).parent.parent.absolute())+'/train_cf_3.npy', allow_pickle=True)
    trainNum = len(train_set.item()['states'])
    print('Number of training samples:', trainNum)

    # load testing dataset
    test_set = np.load(str(Path(os.getcwd()).parent.parent.absolute())+'/test_cf_3.npy', allow_pickle=True)
    testNum = len(test_set.item()['states'])
    print('Number of testing samples:', testNum)

    # save losses
    loss_matrix = []
    min_loss = [] # save the best model
    steps = 0 # args.num_seed_steps
    learn_steps = 0 # env_args.learn_steps
    epoch = 0 # training episodes
    begin_learn = False

    vali1_spacing_rmspe = []
    vali1_speed_rmspe = []
    vali2_spacing_rmspe = []
    vali2_speed_rmspe = []

    # save training records
    f = open(str(Path(os.getcwd()).absolute())+"/vali_IQ_log.txt", "w")

    while True:
        init_data = np.array(train_set.item()['states'][epoch%trainNum])
        state = env.reset(init_data)

        for episode_step in range(EPISODE_STEPS):
            if steps < args.num_seed_steps:
                action = env.action_space.sample()  # Sample random action
            else:
                with eval_mode(agent):
                    # next_states
                    a = []
                    for item in state:
                        if isinstance(item, np.ndarray):
                            a.append(item[0])
                        else:
                            a.append(item)
                    state = np.array(a)
                    action = agent.choose_action(state, sample=True) # Choose action

            next_state, reward, done, _ = env.step(action)
            steps += 1

            # allow infinite bootstrap
            done_no_lim = done
            if str(env.__class__.__name__).find('TimeLimit') >= 0 and episode_step + 1 == EPISODE_STEPS:
                done_no_lim = 0
            online_memory_replay.add((state, next_state, action, reward, done_no_lim))

            if online_memory_replay.size() > INITIAL_MEMORY:
                if begin_learn is False:
                    print('learn begin!')
                    begin_learn = True

                learn_steps += 1
                if epoch > LEARN_EPISODES:
                    """save training losses"""
                    loss_matrix = pd.DataFrame(loss_matrix)
                    loss_matrix.to_csv("train_LQ_loss.csv", index=False)
                    """visualize training losses"""
                    plt.plot(loss_matrix.iloc[:, 0], label=loss_matrix.columns[0])
                    plt.plot(loss_matrix.iloc[:, 1], label=loss_matrix.columns[1])
                    plt.plot(loss_matrix.iloc[:, 2], label=loss_matrix.columns[2])
                    plt.plot(loss_matrix.iloc[:, 3], label=loss_matrix.columns[3])
                    plt.plot(loss_matrix.iloc[:, 4], label=loss_matrix.columns[4])
                    plt.plot(loss_matrix.iloc[:, 5], label=loss_matrix.columns[5])
                    plt.plot(loss_matrix.iloc[:, 6], label=loss_matrix.columns[6])
                    plt.legend()
                    plt.ylim(-2, 2)
                    plt.show()

                    """visualize validation-training and testing logs"""
                    # plt.plot(vali1_spacing_rmspe, label='Train Spacing RMSPE')
                    # plt.plot(vali1_speed_rmspe, label='Train Speed RMSPE')
                    plt.plot(vali2_spacing_rmspe, label='Test Spacing RMSPE')
                    plt.plot(vali2_speed_rmspe, label='Test Speed RMSPE')
                    plt.legend()
                    plt.ylim(0, 1)
                    plt.show()
                    f.close()

                    print('Finished!')
                    return

                ######
                # IRL Modification
                agent.irl_update = types.MethodType(irl_update, agent)
                agent.irl_update_critic = types.MethodType(irl_update_critic, agent)
                losses = agent.irl_update(online_memory_replay,
                                          expert_memory_replay, logger, learn_steps)
                # collect losses
                loss_matrix.append(losses)
                ######

                if learn_steps % args.log_interval == 0:
                    # if args.method.regularize:
                    #     print(
                    #         'TRAIN\tStep {}\tTotal Losses: {:.6f}\tValue Losses: {:.6f}\tRegularize Losses: {:.6f}\tV0 Losses: {:.6f}'.format(
                    #             learn_steps, losses['total_loss'], losses['value_loss'], losses['regularize_loss'], losses['v0']))
                    # else:
                    #     print(
                    #         'TRAIN\tStep {}\tTotal Losses: {:.6f}\tValue Losses: {:.6f}\tchi2 Losses: {:.6f}\tgp loss: {:.6f}\tV0 Losses: {:.6f}'.format(
                    #             learn_steps, losses['total_loss'], losses['value_loss'], losses['chi2_loss'], losses['gp_loss'],
                    #             losses['v0']))

                    for key, loss in losses.items():
                        writer.add_scalar(key, loss, global_step=learn_steps)

            state = next_state
            if done:
                break

        if begin_learn:
            """evaluate the trained model"""
            # # training
            # temp1, temp2 = vali(agent, env, train_set)
            # vali1_spacing_rmspe.append(temp1)
            # vali1_speed_rmspe.append(temp2)
            # testing
            temp3, temp4 = vali(agent, env, test_set)
            vali2_spacing_rmspe.append(temp3)
            vali2_speed_rmspe.append(temp4)

            print(
                'Epoch {}\tValidation-Testing\tSpacing RMSPE: {:.6f}\tSpeed RMSPE: {:.6f}'.format(
                    epoch, temp3, temp4))
            f.write(
                'Epoch {}\tValidation-Testing\tSpacing RMSPE: {:.6f}\tSpeed RMSPE: {:.6f}'.format(
                    epoch, temp3, temp4))
            f.write("\n")

            # save the best model
            if len(min_loss) == 0:
                min_loss.append(vali2_spacing_rmspe[-1])
            else:
                if min_loss[-1] < vali2_spacing_rmspe[-1]:
                    pass
                else:
                    min_loss[-1] = vali2_spacing_rmspe[-1]
                    # save the best trained model
                    save(agent, args, output_dir='results')

        epoch += 1


def cal_rmspe(y_true, y_pred):
    # Compute Root Mean Square Percentage Error between two arrays
    loss = np.sqrt(np.sum(np.square(y_true - y_pred))/np.sum(np.square(y_true)))
    return loss


def save(agent, args, output_dir='results'):
    if args.method.type == "sqil":
        name = f'sqil_{args.env.name}'
    else:
        name = f'iq_{args.env.name}'

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    agent.save(f'{output_dir}/{args.agent.name}_{name}')


def vali(agent, env, data):
    # store simulated and real data
    # spacing
    SimSpaceData = []
    RealSpaceData = []
    # following speed
    SimSpeedData = []
    RealSpeedData = []

    for epoch in range(len(data.item()['states'])):
        init_data = np.array(data.item()['states'][epoch])
        state = env.reset(init_data)

        # store simulated spacing and following speed
        SimSpaceData.append(state[0])
        SimSpeedData.append(state[1])
        # store real spacing and following speed
        RealSpaceData.append(env.RealSpaceData[0])
        RealSpeedData.append(env.RealSpeedData[0])

        while True:
            action = agent.choose_action(state, sample=False)
            next_state, reward, done, _ = env.step(action)

            # next_states
            a = []
            for item in next_state:
                if isinstance(item, np.ndarray):
                    a.append(item[0])
                else:
                    a.append(item)
            next_state = np.array(a)
            state = next_state

            if done:
                break

            # store simulated spacing and following speed
            SimSpaceData.append(state[0])
            SimSpeedData.append(state[1])
            # store real spacing and following speed
            RealSpaceData.append(env.RealSpaceData[env.timeStep - 1])
            RealSpeedData.append(env.RealSpeedData[env.timeStep - 1])

    # spacing
    SimSpaceData = np.array(SimSpaceData)
    RealSpaceData = np.array(RealSpaceData)
    spacing_rmspe = cal_rmspe(y_true=RealSpaceData, y_pred=SimSpaceData)
    # following speed
    SimSpeedData = np.array(SimSpeedData)
    RealSpeedData = np.array(RealSpeedData)
    speed_rmspe = cal_rmspe(y_true=RealSpeedData, y_pred=SimSpeedData)

    return spacing_rmspe, speed_rmspe


"""
IQ-Learn. Do not modify
"""
# Minimal IQ-Learn objective
def iq_learn_update(self, policy_batch, expert_batch, logger, step):
    args = self.args
    policy_obs, policy_next_obs, policy_action, policy_reward, policy_done = policy_batch
    expert_obs, expert_next_obs, expert_action, expert_reward, expert_done = expert_batch

    if args.only_expert_states:
        expert_batch = expert_obs, expert_next_obs, policy_action, expert_reward, expert_done

    obs, next_obs, action, reward, done, is_expert = get_concat_samples(
        policy_batch, expert_batch, args)

    if self.actor:
        policy_next_actions, policy_log_prob, _ = self.actor.sample(policy_next_obs)

    losses = {}

    ######
    # IQ-Learn minimal implementation with X^2 divergence (~15 lines)
    # Calculate 1st term of loss: -E_(ρ_expert)[Q(s, a) - γV(s')]
    current_Q = self.critic(obs, action)
    y = (1 - done) * self.gamma * self.getV(next_obs)
    if args.train.use_target:
        with torch.no_grad():
            y = (1 - done) * self.gamma * self.get_targetV(next_obs)

    reward = (current_Q - y)[is_expert]
    loss = -(reward).mean()

    # 2nd term for our loss (use expert and policy states): E_(ρ)[Q(s,a) - γV(s')]
    value_loss = (self.getV(obs) - y).mean()
    loss += value_loss

    # Use χ2 divergence (adds an extra term to the loss)
    chi2_loss = 1/(4 * args.method.alpha) * (reward**2).mean()
    loss += chi2_loss
    ######

    self.critic_optimizer.zero_grad()
    loss.backward()
    self.critic_optimizer.step()
    return loss


# Full IQ-Learn objective with other divergences and options
def irl_update_critic(self, policy_batch, expert_batch, logger, step):
    args = self.args
    policy_obs, policy_next_obs, policy_action, policy_reward, policy_done = policy_batch
    expert_obs, expert_next_obs, expert_action, expert_reward, expert_done = expert_batch

    if args.only_expert_states:
        # Use policy actions instead of experts actions for IL with only observations
        expert_batch = expert_obs, expert_next_obs, policy_action, expert_reward, expert_done

    obs, next_obs, action, reward, done, is_expert = get_concat_samples(
        policy_batch, expert_batch, args)

    losses = {}
    # keep track of v0
    v0 = self.getV(expert_obs[:, :, -1]).mean()
    losses['v0'] = v0.item()

    if args.method.type == "sqil":
        with torch.no_grad():
            target_Q = reward + (1 - done) * self.gamma * self.get_targetV(next_obs)

        current_Q = self.critic(obs, action)
        bell_error = F.mse_loss(current_Q, target_Q, reduction='none')
        loss = (bell_error[is_expert]).mean() + \
            args.method.sqil_lmbda * (bell_error[~is_expert]).mean()
        losses['sqil_loss'] = loss.item()

    elif args.method.type == "iq":
        # our method, calculate 1st term of loss
        #  -E_(ρ_expert)[Q(s, a) - γV(s')]
        current_Q = self.critic(obs[:, :, -1], action)
        next_v = self.getV(next_obs[:, :, -1])
        y = (1 - done) * self.gamma * next_v

        if args.train.use_target:
            with torch.no_grad():
                next_v = self.get_targetV(next_obs)
                y = (1 - done) * self.gamma * next_v

        reward = (current_Q - y)[is_expert]

        with torch.no_grad():
            if args.method.div == "hellinger":
                phi_grad = 1/(1+reward)**2
            elif args.method.div == "kl":
                phi_grad = torch.exp(-reward-1)
            elif args.method.div == "kl2":
                phi_grad = F.softmax(-reward, dim=0) * reward.shape[0]
            elif args.method.div == "kl_fix":
                phi_grad = torch.exp(-reward)
            elif args.method.div == "js":
                phi_grad = torch.exp(-reward)/(2 - torch.exp(-reward))
            else:
                phi_grad = 1
        loss = -(phi_grad * reward).mean()
        losses['softq_loss'] = loss.item()

        if args.method.loss == "v0":
            # calculate 2nd term for our loss
            # (1-γ)E_(ρ0)[V(s0)]
            v0_loss = (1 - self.gamma) * v0
            loss += v0_loss
            losses['v0_loss'] = v0_loss.item()

        elif args.method.loss == "value":
            # alternative 2nd term for our loss (use expert and policy states)
            # E_(ρ)[Q(s,a) - γV(s')]
            value_loss = (self.getV(obs[:, :, -1]) - y).mean()
            loss += value_loss
            losses['value_loss'] = value_loss.item()

        elif args.method.loss == "value_policy":
            # alternative 2nd term for our loss (use only policy states)
            # E_(ρ)[Q(s,a) - γV(s')]
            value_loss = (self.getV(obs) - y)[~is_expert].mean()
            loss += value_loss
            losses['value_policy_loss'] = value_loss.item()

        elif args.method.loss == "value_expert":
            # alternative 2nd term for our loss (use only expert states)
            # E_(ρ)[Q(s,a) - γV(s')]
            value_loss = (self.getV(obs) - y)[is_expert].mean()
            loss += value_loss
            losses['value_loss'] = value_loss.item()

        elif args.method.loss == "value_mix":
            # alternative 2nd term for our loss (use expert and policy states)
            # E_(ρ)[Q(s,a) - γV(s')]
            w = args.method.mix_coeff
            value_loss = (w * (self.getV(obs) - y)[is_expert] +
                          (1-w) * (self.getV(obs) - y)[~is_expert]).mean()
            loss += value_loss
            losses['value_loss'] = value_loss.item()

        elif args.method.loss == "skip":
            # No loss
            pass
    else:
        raise ValueError(f'This method is not implemented: {args.method.type}')

    if args.method.grad_pen:
        # add a gradient penalty to loss (W1 metric)
        gp_loss = self.critic_net.grad_pen(expert_obs, expert_action,
                                           policy_obs, policy_action, args.method.lambda_gp)
        losses['gp_loss'] = gp_loss.item()
        loss += gp_loss

    if args.method.div == "chi" or args.method.chi:
        # Use χ2 divergence (adds an extra term to the loss)
        if args.train.use_target:
            with torch.no_grad():
                next_v = self.get_targetV(next_obs[:, :, -1])
        else:
            next_v = self.getV(next_obs[:, :, -1])

        y = (1 - done) * self.gamma * next_v

        current_Q = self.critic(obs[:, :, -1], action)
        reward = current_Q - y
        chi2_loss = 1/(4 * args.method.alpha) * (reward**2)[is_expert].mean()
        loss += chi2_loss
        losses['chi2_loss'] = chi2_loss.item()

    if args.method.regularize:
        # Use χ2 divergence (adds an extra term to the loss)
        if args.train.use_target:
            with torch.no_grad():
                next_v = self.get_targetV(next_obs[:, :, -1])
        else:
            next_v = self.getV(next_obs[:, :, -1])

        y = (1 - done) * self.gamma * next_v

        current_Q = self.critic(obs[:, :, -1], action)
        reward = current_Q - y
        chi2_loss = 1/(4 * args.method.alpha) * (reward**2).mean()
        loss += chi2_loss
        losses['regularize_loss'] = chi2_loss.item()

    losses['total_loss'] = loss.item()

    # Optimize the critic
    self.critic_optimizer.zero_grad()
    loss.backward()
    # step critic
    self.critic_optimizer.step()
    return losses


def irl_update(self, policy_buffer, expert_buffer, logger, step):
    policy_batch = policy_buffer.get_samples(self.batch_size, self.device)
    expert_batch = expert_buffer.get_samples(self.batch_size, self.device)

    losses = self.irl_update_critic(policy_batch, expert_batch, logger, step)

    if self.actor and step % self.actor_update_frequency == 0:
        if not self.args.agent.vdice_actor:

            if self.args.offline:
                obs = expert_batch[0]
            else:
                # Use both policy and expert observations
                obs = torch.cat([policy_batch[0], expert_batch[0]], dim=0)

            if self.args.num_actor_updates:
                for i in range(self.args.num_actor_updates):
                    actor_alpha_losses = self.update_actor_and_alpha(obs[:, :, -1], logger, step)

            # actor_alpha_losses = self.update_actor_and_alpha(obs, logger, step)
            losses.update(actor_alpha_losses)

    if step % self.critic_target_update_frequency == 0:
        if self.args.train.soft_update:
            soft_update(self.critic_net, self.critic_target_net,
                        self.critic_tau)
        else:
            hard_update(self.critic_net, self.critic_target_net)
    return losses


if __name__ == "__main__":
    main()