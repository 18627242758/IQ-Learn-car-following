# IQ-Learn-car-following
This repo implements Inverse Q-Learning (IQ-Learn) to reproduce human-driven vehicles' trajectories when they are following autonomous vehicles using the Waymo Open Dataset.
# Description
The IQ-Learn method is adapted from https://github.com/Div99/IQ-Learn. In this study, it is used to mimic human driver's trajectories when they are following autonomous vehicles using the Waymo Open Dataset. There are also some benchmark methods including the intelligent driver model (IDM), long short-term memory (LSTM) neural network, generative adversarial imitation learning (GAIL) and adversarial inverse reinforcement learning (AIRL).
# Data format
Each folder represents a car-following style in different car-following modes. Each element in the train_cf_*.mat, train_cf_*.npy, test_cf_*.mat, and test_cf_*.npy describes a car-following event. Training and testing sets are saved in the following manner: [‘states’, ‘next_states’, ‘actions’, ‘rewards’, ‘dones’, ‘lengths’]. For each state, the columns are spacing, following vehicle speed, relative speed (following vehicle speed - leading vehicle speed), and leading vehicle speed. Events may have different durations.
# How to run
For IQ-Learn, please run train_IQ.py to train the IQ-Learn model using the training set and test_IQ.py to validate on the testing set. For benchmarks methods, please run the corresponding .ipynb file.
