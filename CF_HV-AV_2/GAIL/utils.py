import os
import numpy as np
import torch
from itertools import count
import matplotlib.pyplot as plt


def soft_update(target, source, tau):
    """Soft update for SAC"""
    for t, s in zip(target.parameters(), source.parameters()):
        t.data.mul_(1.0 - tau)
        t.data.add_(tau * s.data)


def disable_gradient(network):
    """Disable the gradients of parameters in the network"""
    for param in network.parameters():
        param.requires_grad = False


def add_random_noise(action, std):
    """Add random noise to the action"""
    action += np.random.randn(*action.shape) * std
    return action.clip(-1.0, 1.0)


def cal_rmspe(y_true, y_pred):
    # Compute Root Mean Square Percentage Error between two arrays
    loss = np.sqrt(np.sum(np.square(y_true - y_pred))/np.sum(np.square(y_true)))
    return loss


def evaluation(name, env, algo, buffer, seed: int = 0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # store the simulated trajectories
    sim_traj = []

    # load testing dataset
    test_set = np.load(buffer, allow_pickle=True)
    testNum = len(test_set.item()['states'])
    print('Number of testing samples:', testNum)

    # store simulated and real data
    # spacing
    SimSpaceData = []
    RealSpaceData = []
    # following speed
    SimSpeedData = []
    RealSpeedData = []

    for epoch in range(testNum):
        # Initialize the environment.
        init_data = np.array(test_set.item()['states'][epoch])
        state = env.reset(init_data)
        traj = [state]

        # store simulated spacing and following speed
        SimSpaceData.append(state[0])
        SimSpeedData.append(state[1])

        # store real spacing and following speed
        RealSpaceData.append(env.RealSpaceData[0])
        RealSpeedData.append(env.RealSpeedData[0])

        for _ in count():
            action = algo.exploit(state)
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
        sim_traj.append(traj)

    # spacing
    SimSpaceData = np.array(SimSpaceData)
    RealSpaceData = np.array(RealSpaceData)
    spacing_rmspe = cal_rmspe(y_true=RealSpaceData, y_pred=SimSpaceData)

    # following speed
    SimSpeedData = np.array(SimSpeedData)
    RealSpeedData = np.array(RealSpeedData)
    speed_rmspe = cal_rmspe(y_true=RealSpeedData, y_pred=SimSpeedData)

    print(
        'TEST\tTotal\tAverage Spacing RMSPE: {:.6f}\tAverage Speed RMSPE: {:.6f}'.format(
            spacing_rmspe, speed_rmspe))

    np.save("result/test_"+name+"_traj.npy", sim_traj)


def visualize(name, buffer, traj_path, num):
    # load testing dataset
    test_set = np.load(buffer, allow_pickle=True)
    test_set = test_set.item()['states']
    test_set = np.array([np.array(x) for x in test_set])
    # load simulated dataset
    test_traj = np.load(traj_path+"test_"+name+"_traj.npy", allow_pickle=True)

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