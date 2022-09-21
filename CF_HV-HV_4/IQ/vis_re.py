import os
import sys
from pathlib import Path
import pandas as pd
from omegaconf import DictConfig, OmegaConf
import torch
import hydra
from itertools import count
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

sys.path.append(str(Path(os.getcwd()).parent.parent.absolute()))
from CF_Env import CF_Env
from agent import sac


def get_args(cfg: DictConfig):
    cfg.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # print(OmegaConf.to_yaml(cfg))
    return cfg


@hydra.main(config_path="./conf", config_name="config")
def main(cfg: DictConfig):
    args = get_args(cfg)

    GAMMA = args.gamma
    grid_size = 0.05

    # load the environment
    env = CF_Env()

    # load the agent
    agent = sac.SAC(env.observation_space.shape[0], env.action_space.shape[0],
                    [env.action_LowerBound, env.action_HigherBound],
                    args.train.batch, args)

    # load the policy
    if args.method.type == "sqil":
        name = f'sqil'
    else:
        name = f'iq'

    policy_file = f'outputs/results'

    if args.eval.policy:
        policy_file = f'{args.eval.policy}'
    print(f'Loading policy from: {policy_file}')

    if args.eval.transfer:
        agent.load(hydra.utils.to_absolute_path(policy_file),
                   f'_{name}_{args.eval.expert_env}')
    else:
        agent.load(hydra.utils.to_absolute_path(policy_file), f'_{name}_{args.env.name}')

    # load testing dataset
    test_set = np.load(str(Path(os.getcwd()).parent.parent.parent.parent.absolute())+'/test_cf_4.npy', allow_pickle=True)
    testNum = len(test_set.item()['states'])
    print('Number of testing samples:', testNum)

    # visualize rewards
    # reward functions are presented as bivariate state feature spaces, while holding the other states at their mean values
    visualize_reward_total(agent, env, test_set, args, grid_size)

    for epoch in range(testNum):
        # initialize the environment
        init_data = np.array(test_set.item()['states'][epoch])
        state = env.reset(init_data)

        # cumulative rewards
        episode_irl_reward = 0

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

            # Get sqil reward
            with torch.no_grad():
                q = agent.infer_q(state, action)
                next_v = agent.infer_v(next_state)
                y = (1 - done) * GAMMA * next_v
                irl_reward = q - y

            episode_irl_reward += irl_reward.item()
            state = next_state

            if done:
                break

        print('Ep {}\tMoving Soft Q average rewards: {:.4f}\t'.format(epoch, episode_irl_reward))


def visualize_reward_total(agent, env, data, args, grid_size):
    # make states as dataframe
    new_states = []
    states = data.item()['states']

    for items in states:
        for item in items:
            new_states.append(item)
        new_states = pd.DataFrame(new_states, columns=['Spacing (m)', 'Following Vehicle Speed (m/s)', 'Relative Speed (m/s)', 'Leading Vehicle Speed (m/s)'])

    # visualize the rewards
    for i in range(new_states.shape[1]-2):
        for j in range(i+1, new_states.shape[1]-1):
            # store the mean values
            state = new_states.mean(axis=0)

            # generate auxiliary points
            total_rewards = []

            if new_states.columns[i] == 'Spacing (m)':
                new_i = np.linspace(5, 30,
                                    num=round((np.max(new_states.iloc[:, i]) - np.min(new_states.iloc[:, i])) / grid_size),
                                    endpoint=True)
            elif new_states.columns[i] == 'Following Vehicle Speed (m/s)':
                new_i = np.linspace(3, 16,
                                    num=round((np.max(new_states.iloc[:, i]) - np.min(new_states.iloc[:, i])) / grid_size),
                                    endpoint=True)
            elif new_states.columns[i] == 'Relative Speed (m/s)':
                new_i = np.linspace(-2, 2,
                                    num=round((np.max(new_states.iloc[:, i]) - np.min(new_states.iloc[:, i])) / grid_size),
                                    endpoint=True)

            if new_states.columns[j] == 'Spacing (m)':
                new_j = np.linspace(5, 30,
                                    num=round((np.max(new_states.iloc[:, j]) - np.min(new_states.iloc[:, j])) / grid_size),
                                    endpoint=True)
            elif new_states.columns[j] == 'Following Vehicle Speed (m/s)':
                new_j = np.linspace(3, 16,
                                    num=round((np.max(new_states.iloc[:, j]) - np.min(new_states.iloc[:, j])) / grid_size),
                                    endpoint=True)
            elif new_states.columns[j] == 'Relative Speed (m/s)':
                new_j = np.linspace(-2, 2,
                                    num=round((np.max(new_states.iloc[:, j]) - np.min(new_states.iloc[:, j])) / grid_size),
                                    endpoint=True)

            # save the rewards
            for k in range(len(new_i)):
                rewards = []
                # update state
                state[i] = new_i[k]

                for l in range(len(new_j)):
                    # update state
                    state[j] = new_j[l]

                    init_state = env.reset(np.array([[x] for x in state]).reshape(1, -1))
                    action = agent.choose_action(init_state, sample=False)

                    # modify the environment
                    state = np.array(state)
                    env.LVSpdData = np.append(env.LVSpdData, np.mean(state[3]))
                    next_state, reward, done, _ = env.step(action)

                    # next_states
                    a = []
                    for item in next_state:
                        if isinstance(item, np.ndarray):
                            a.append(item[0])
                        else:
                            a.append(item)
                    next_state = np.array(a)

                    with torch.no_grad():
                        state = torch.FloatTensor(state).to(agent.device)
                        action = torch.FloatTensor(action).to(agent.device)
                        next_state = torch.FloatTensor(next_state).to(agent.device)

                    # Get sqil reward
                    with torch.no_grad():
                        q = agent.critic(state[:3], action)

                        next_v = agent.getV(next_state)
                        y = (1 - done) * args.gamma * next_v
                        irl_reward = q - y

                        # irl_reward = -irl_reward.cpu().numpy() # TODO: why negative? https://github.com/Div99/IQ-Learn/blob/main/iq_learn/vis/maze_vis.py
                        irl_reward = irl_reward.cpu().numpy()

                    rewards.append(irl_reward[0])

                total_rewards.append(rewards)

            # draw recovered reward plots
            plt.figure(figsize=(8,6), dpi=300)
            yi, xi = np.meshgrid(new_j, new_i)
            plt.pcolormesh(xi, yi, np.array(total_rewards).reshape(xi.shape), shading='auto', cmap='viridis')
            plt.xlabel(new_states.columns[i], size=20)
            plt.xticks(size=16)
            plt.ylabel(new_states.columns[j], size=20)
            plt.yticks(size=16)
            clb = plt.colorbar()
            clb.ax.set_title('Reward', size=20)
            clb.ax.tick_params(labelsize=16)

            # save plots
            plt.savefig(str(Path(os.getcwd()).parent.parent.absolute())+'/'+new_states.columns[i].split()[0]+'_'+new_states.columns[j].split()[0]+'.png')
            print('Save feature: \tX axis {}\t_\tY axis {}'.format(new_states.columns[i], new_states.columns[j]))
            plt.close()


if __name__ == '__main__':
    main()