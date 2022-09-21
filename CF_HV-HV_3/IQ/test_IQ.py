import hydra
import torch
import numpy as np

import os
import sys
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
sys.path.append(str(Path(os.getcwd()).parent.parent.absolute()))
from CF_Env import CF_Env
from agent import sac
import matplotlib.pyplot as plt


def get_args(cfg: DictConfig):
    cfg.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # print(OmegaConf.to_yaml(cfg))
    return cfg


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    args = get_args(cfg)

    # load the environment
    env = CF_Env()

    # load the agent
    agent = sac.SAC(env.observation_space.shape[0], env.action_space.shape[0],
                    [env.action_LowerBound, env.action_HigherBound],
                    args.train.batch, args)

    if args.method.type == "sqil":
        name = f'sqil'
    else:
        name = f'iq'

    # load the trained policy
    policy_file = f'{str(Path(os.getcwd()).parent.parent.absolute())}'+'/results/'
    if args.eval.policy:
        policy_file = f'{args.eval.policy}'
    print(f'Loading policy from: {policy_file}')

    if args.eval.transfer:
        agent.load(hydra.utils.to_absolute_path(policy_file),
                   f'_{name}_{args.eval.expert_env}')
    else:
        agent.load(hydra.utils.to_absolute_path(policy_file), f'_{name}_{args.env.name}')

    if args.eval_only:
        exit()

    # fit on the testing set
    predict_testset(agent, env, args, log=True)

    # visualize the real and simulated trajectories
    vis_traj(num=0)

def predict_testset(agent, env, args, log=True):
    # store the simulated trajectories
    iq_test = []

    # load testing dataset
    test_set = np.load(str(Path(os.getcwd()).parent.parent.parent.parent.absolute())+'/test_cf_3.npy', allow_pickle=True)
    testNum = len(test_set.item()['states'])
    print('Number of testing samples:', testNum)

    GAMMA = args.gamma

    # store simulated and real data
    # spacing
    SimSpaceData = []
    RealSpaceData = []
    # following speed
    SimSpeedData = []
    RealSpeedData = []

    for epoch in range(testNum):
        init_data = np.array(test_set.item()['states'][epoch])
        state = env.reset(init_data)
        traj = [state]
        episode_irl_reward = 0

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

            # Get sqil reward
            with torch.no_grad():
                q = agent.infer_q(state, action)
                next_v = agent.infer_v(next_state)
                y = (1 - done) * GAMMA * next_v
                irl_reward = (q - y)

            episode_irl_reward += irl_reward.item()

            state = next_state
            traj.append(state)

            if done:
                break

            # store simulated spacing and following speed
            SimSpaceData.append(state[0])
            SimSpeedData.append(state[1])
            # store real spacing and following speed
            RealSpaceData.append(env.RealSpaceData[env.timeStep - 1])
            RealSpeedData.append(env.RealSpeedData[env.timeStep - 1])

        traj = np.array(traj)
        iq_test.append(traj)

        if log:
            print('Ep {}\tEpisode learnt rewards {:.2f}\t'.format(epoch, episode_irl_reward))

    # spacing
    SimSpaceData = np.array(SimSpaceData)
    RealSpaceData = np.array(RealSpaceData)
    spacing_rmspe = cal_rmspe(y_true=RealSpaceData, y_pred=SimSpaceData)
    # following speed
    SimSpeedData = np.array(SimSpeedData)
    RealSpeedData = np.array(RealSpeedData)
    speed_rmspe = cal_rmspe(y_true=RealSpeedData, y_pred=SimSpeedData)
    print(
        'TEST\tSpacing RMSPE: {:.6f}\tSpeed RMSPE: {:.6f}'.format(
            spacing_rmspe, speed_rmspe))

    np.save(str(Path(os.getcwd()).parent.parent.absolute())+'/test_IQ_traj.npy', iq_test)


def vis_traj(num=0):
    # real data: spacing, relative speed and following vehicle speed
    test_set = np.load(str(Path(os.getcwd()).parent.parent.parent.parent.absolute())+'/test_cf_3.npy',
                       allow_pickle=True)
    test_set = test_set.item()['states']
    test_set = np.array([np.array(x) for x in test_set])
    # simulated data: spacing, relative speed and following vehicle speed
    test_traj = np.load(str(Path(os.getcwd()).parent.parent.absolute())+'/test_IQ_traj.npy',
                        allow_pickle=True)

    # spacing
    plt.plot(test_traj[num][:, 0], label='Sim Spacing')
    plt.plot(test_set[num][:, 0], label='Real Spacing')
    plt.legend()
    plt.show()

    # following speed
    plt.plot(test_traj[num][:, 1], label='Sim FV Speed')
    plt.plot(test_set[num][:, 1], label='Real FV Speed')
    plt.legend()
    plt.show()


def cal_rmspe(y_true, y_pred):
    # Compute Root Mean Square Percentage Error between two arrays
    loss = np.sqrt(np.sum(np.square(y_true - y_pred))/np.sum(np.square(y_true)))
    return loss


def eps(rewards):
    return [sum(x) for x in rewards]


def part_eps(rewards):
    return [np.cumsum(x) for x in rewards]


if __name__ == '__main__':
    main()