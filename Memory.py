import random
from collections import deque, namedtuple

import numpy as np
import torch


class Memory:
    def __init__(self, max_size, device):
        self.buffer = deque(maxlen=max_size)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.device = device

    def add(self, state, action, reward, next_state, done):
        experience = self.experience(state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        experiences = random.sample(self.buffer, batch_size)
        minibatch = np.asarray(experiences)

        # Convert to torch tensors
        states = torch.from_numpy(np.vstack(minibatch[:, 0])).float().to(self.device)
        actions = torch.from_numpy(np.vstack(minibatch[:, 1])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack(minibatch[:, 2])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack(minibatch[:, 3])).float().to(self.device)
        dones = torch.from_numpy(
            np.vstack([experience.done for experience in experiences if experience is not None]).astype(np.uint8)).float().to(self.device)

        return states, actions, rewards, next_states, dones

    def size(self):
        return len(self.buffer)
