import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from Memory import Memory
from QNetwork import QNetwork

plt.style.use('seaborn-whitegrid')


class DQNetwork:
    """ Implementation of deep q learning algorithm """

    def __init__(self, state_space, action_space, learning_rate, memory_size, batch_size, gamma, epsilon,
                 epsilon_decay, epsilon_min, min_samples, seed, training=True):
        # if gpu is to be used
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'Using {self.device} for computing')
        self.action_space = action_space
        self.state_space = state_space
        self.lr = learning_rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.gamma = gamma
        self.min_samples = min_samples
        self.started_learning = False
        self.is_training = training
        self.seed = seed

        # Initialize model
        self.q_network = QNetwork(state_space, action_space, seed).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        if training:
            # Initialize memory
            self.memory = Memory(max_size=memory_size, device=self.device)

    def get_model(self):
        return self.q_network

    def set_model(self, q_network):
        self.q_network = q_network

    # add experience to agent's memory
    def add_to_memory(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

    def replay(self, poison_data=False):
        if self.memory.size() < self.min_samples:
            return
        self.started_learning = True
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        # Get the action with max Q value
        action_values = self.q_network(next_states).detach()
        # unsqueeze operation --> np.reshape
        # Here, we make it from torch.Size([4]) -> torch.Size([4, 1])
        max_action_values = action_values.max(1)[0].unsqueeze(1)

        # If done just use reward, else update Q_target with discounted action values
        q_targets = rewards + (self.gamma * max_action_values * (1 - dones))
        q_expected = self.q_network(states).gather(1, actions)

        # Calculate loss
        loss = F.mse_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        # backward pass
        loss.backward()
        # update weights
        self.optimizer.step()

        # update exploration/exploitation balance
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min

    # With ϵ select a random action at, otherwise select at=argmaxaQ(st,a)
    def predict_action(self, state):
        if ((np.random.rand() <= self.epsilon) or (not self.started_learning)) and self.is_training:
            return random.randrange(self.action_space)
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        # set the network into evaluation mode
        self.q_network.eval()
        with torch.no_grad():
            action_values = self.q_network(state)
        # Back to training mode
        self.q_network.train()
        action = np.argmax(action_values.cpu().data.numpy())
        return action

    def checkpoint(self, filename):
        torch.save(self.q_network.state_dict(), filename)
