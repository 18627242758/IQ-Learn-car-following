from collections import deque
import numpy as np
import random
import torch
from expert_dataset import ExpertDataset


class Memory(object):
    def __init__(self, memory_size: int, seed: int = 0) -> None:
        random.seed(seed)
        self.memory_size = memory_size
        self.buffer = deque(maxlen=self.memory_size)

    def add(self, experience) -> None:
        self.buffer.append(experience)

    def size(self):
        return len(self.buffer)

    def sample(self, batch_size: int, continuous: bool = True):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)

        if continuous:
            rand = random.randint(0, len(self.buffer) - batch_size)
            return [self.buffer[i] for i in range(rand, rand + batch_size)]
        else:
            indexes = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
            return [self.buffer[i] for i in indexes]

    def clear(self):
        self.buffer.clear()

    def save(self, path):
        b = np.asarray(self.buffer)
        print(b.shape)
        np.save(path, b)

    def load(self, path, num_trajs, sample_freq, seed):
        data = ExpertDataset(path, num_trajs, sample_freq, seed)
        # data = np.load(path, allow_pickle=True)
        for i in range(len(data)):
            self.add(data[i])

    def get_samples(self, batch_size, device):
        batch = self.sample(batch_size, False)

        batch_state, batch_next_state, batch_action, batch_reward, batch_done = zip(
            *batch)

        batch_state = np.array(batch_state)
        batch_next_state = np.array(batch_next_state)

        # batch_state
        a = []
        b = []
        for items in batch_state:
            for item in items:
                a.append(item)
            b.append(np.array(a)[:3].reshape(-1, 1))
            a = []
        batch_state = np.array(b)

        # batch_next_state
        a = []
        b = []
        for items in batch_next_state:
            for item in items:
                if isinstance(item, float) or isinstance(item, int):
                    a.append(item)
                else:
                    a.append(item[0])
            b.append(np.array(a)[:3].reshape(-1, 1))
            a = []
        batch_next_state = np.array(b)

        # batch_action
        a = []
        for item in batch_action:
            if isinstance(item, str):
                a.append(float(item))
            else:
                a.append(item[0])
        batch_action = np.array(a)

        batch_state = torch.as_tensor(batch_state, dtype=torch.float, device=device)
        batch_next_state = torch.as_tensor(batch_next_state, dtype=torch.float, device=device)
        batch_action = torch.as_tensor(batch_action, dtype=torch.float, device=device)

        if batch_action.ndim == 1:
            batch_action = batch_action.unsqueeze(1)
        batch_reward = torch.as_tensor(batch_reward, dtype=torch.float, device=device).unsqueeze(1)
        batch_done = torch.as_tensor(batch_done, dtype=torch.float, device=device).unsqueeze(1)

        return batch_state, batch_next_state, batch_action, batch_reward, batch_done