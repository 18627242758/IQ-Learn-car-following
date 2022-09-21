import os
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt


class Trainer:
    """
    Trainer for all the algorithms
    Parameters
    ----------
    env: Environment
        environment for training
    algo: Algorithm
        the algorithm to be trained
    log_dir: str
        path to save logs
    buffer_path: str
        path to the demonstration buffer
    num_steps: int
        number of steps to train
    valid_path: str
        path to the validation buffer
    """
    def __init__(self, env, algo, log_dir, buffer_path, num_steps, valid_path):
        super().__init__()

        # Env to collect samples.
        self.env = env

        self.algo = algo
        self.log_dir = log_dir

        self.buffer_path = buffer_path
        self.valid_path = valid_path

        # Log setting.
        self.summary_dir = os.path.join(log_dir, 'summary')
        self.writer = SummaryWriter(log_dir=self.summary_dir)
        self.model_dir = os.path.join(log_dir, 'model')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        # Other parameters.
        self.num_steps = num_steps

    def train(self):
        start_time = time.time()
        print('learn begin!')

        # Episode's timestep.
        t = 0
        # Number of episodes
        episode = 0
        # validation loss
        min_loss = []
        vali1_spacing_rmspe = []
        vali1_speed_rmspe = []

        vali2_spacing_rmspe = []
        vali2_speed_rmspe = []

        global train_set, trainNum, test_set, testNum

        # load training dataset
        train_set = np.load(self.buffer_path, allow_pickle=True)
        trainNum = len(train_set.item()['states'])
        print('Number of training samples:', trainNum)
        # load testing dataset
        test_set = np.load(self.valid_path, allow_pickle=True)
        testNum = len(test_set.item()['states'])
        print('Number of testing samples:', testNum)

        # begin training
        for _ in tqdm(range(self.num_steps)):
            # Initialize the environment.
            init_data = np.array(train_set.item()['states'][episode%trainNum])
            state = self.env.reset(init_data)

            while True:
                # Pass to the algorithm to update state and episode timestep.
                next_state, done, t = self.algo.step(self.env, state, t)

                # Update the algorithm whenever ready.
                if self.algo.is_update(t):
                    self.algo.update(self.writer)

                # check if  training process is done
                if t > self.num_steps:
                    # """visualize validation-training logs"""
                    # plt.plot(vali1_spacing_rmspe, label='Train Spacing RMSPE')
                    # plt.plot(vali1_speed_rmspe, label='Train Speed RMSPE')
                    # plt.legend()
                    # plt.ylim(0, 1)
                    # plt.show()

                    """visualize validation-testing logs"""
                    plt.plot(vali2_spacing_rmspe, label='Test Spacing RMSPE')
                    plt.plot(vali2_speed_rmspe, label='Test Speed RMSPE')
                    plt.legend()
                    plt.ylim(0, 1)
                    plt.show()

                    # Return current training time
                    end_time = time.time()
                    print('Running time:', end_time - start_time)
                    print('Running time per event:', (end_time - start_time) / episode)
                    print('Finished!')

                    return

                state = next_state
                # check if training process for the episode is done
                if done:
                    break

            """evaluate the trained model"""
            # training data
            # temp1, temp2 = self.vali(self.algo, self.env, trainNum, train_set)
            # print(
            #     'Validation-Training\tEpoch {}\tSpacing RMSPE: {:.6f}\tSpeed RMSPE: {:.6f}'.format(
            #         episode, np.mean(temp1), np.mean(temp2)))
            #
            # vali1_spacing_rmspe.append(temp1)
            # vali1_speed_rmspe.append(temp2)

            # tetsing data
            temp1, temp2 = self.vali(self.algo, self.env, testNum, test_set)
            print(
                'Validation-Testing\tEpoch {}\tSpacing RMSPE: {:.6f}\tSpeed RMSPE: {:.6f}'.format(
                    episode, np.mean(temp1), np.mean(temp2)))

            vali2_spacing_rmspe.append(temp1)
            vali2_speed_rmspe.append(temp2)

            # save the best model
            if len(min_loss) == 0:
                min_loss.append(vali2_spacing_rmspe[-1])
            else:
                if min_loss[-1] < vali2_spacing_rmspe[-1]:
                    pass
                else:
                    min_loss[-1] = vali2_spacing_rmspe[-1]
                    # save model
                    self.algo.save_models(os.path.join(self.model_dir))

            episode += 1

    def cal_rmspe(self, y_true, y_pred):
        # Compute Root Mean Square Percentage Error between two arrays
        loss = np.sqrt(np.sum(np.square(y_true - y_pred))/np.sum(np.square(y_true)))
        return loss

    def vali(self, algo, env, num, buffer):
        # store simulated and real data
        # spacing
        SimSpaceData = []
        RealSpaceData = []
        # following speed
        SimSpeedData = []
        RealSpeedData = []

        for epoch in range(num):
            # Initialize the environment.
            init_data = np.array(buffer.item()['states'][epoch])
            state = env.reset(init_data)

            # store simulated spacing and following speed
            SimSpaceData.append(state[0])
            SimSpeedData.append(state[1])
            # store real spacing and following speed
            RealSpaceData.append(env.RealSpaceData[0])
            RealSpeedData.append(env.RealSpeedData[0])

            while True:
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
        spacing_rmspe = self.cal_rmspe(y_true=RealSpaceData, y_pred=SimSpaceData)
        # following speed
        SimSpeedData = np.array(SimSpeedData)
        RealSpeedData = np.array(RealSpeedData)
        speed_rmspe = self.cal_rmspe(y_true=RealSpeedData, y_pred=SimSpeedData)

        return spacing_rmspe, speed_rmspe