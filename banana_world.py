import os
import random as rn
import sys
from collections import deque
from pathlib import Path
import torch

import numpy as np
from unityagents import UnityEnvironment

from DQNetwork import DQNetwork

SEED = 123
# set random seeds
os.environ['PYTHONHASHSEED'] = '0'
rn.seed(SEED)
np.random.seed(SEED)

Path("./results").mkdir(parents=True, exist_ok=True)
Path("./models").mkdir(parents=True, exist_ok=True)

# env = UnityEnvironment(file_name="./Banana_Linux/Banana.x86_64")
env = UnityEnvironment(file_name="./Banana_Linux_NoVis/Banana.x86_64")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# print details about the environment
# number of agents in the environment
print('Number of agents:', len(env_info.agents))

# number of actions
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)

# examine the state space
state = env_info.vector_observations[0]
print('States look like:', state.shape)
state_size = len(state)
print('States have length:', state_size)

# Model hyper-parameters
learning_rate = 0.0001
# Training hyper-parameters
total_episodes = 2000  # Total episodes for training
max_steps = 1000  # Max possible steps in an episode
batch_size = 32  # Batch size
epsilon = 1.0
epsilon_decay = 0.999
epsilon_min = 0.01
# Q learning hyper-parameters
gamma = 0.99  # Discounting rate
memory_size = 10000  # Number of experiences the Memory can keep
min_samples = batch_size
is_training = False
filename = f'{learning_rate}_{gamma}_{epsilon_decay}'

if is_training:
    agent = DQNetwork(state_size, action_size, learning_rate, memory_size, batch_size, gamma, epsilon, epsilon_decay,
                      epsilon_min, min_samples, SEED)
else:
    agent = DQNetwork(state_size, action_size, learning_rate, memory_size, batch_size, gamma, epsilon, epsilon_decay,
                      epsilon_min, min_samples, SEED, training=False)
    agent.q_network.load_state_dict(torch.load(f'./models/trained_agent_{filename}.pt'))

# Train the DQN
last_m_rewards = deque(maxlen=100)
# Cumulative reward
episode_rewards = np.zeros(total_episodes)
rolling_average_rewards = np.zeros(total_episodes)
max_avg_score = -np.inf

for episode in range(total_episodes):

    env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
    state = env_info.vector_observations[0]  # get the current state
    state = state.reshape((1, state_size))
    episode_score = 0  # initialize the score
    for step in range(max_steps):
        # query next action from learner
        action = agent.predict_action(state)
        # perform action

        env_info = env.step(action)[brain_name]  # send the action to the environment
        next_state = env_info.vector_observations[0]  # get the next state
        next_state = next_state.reshape((1, state_size))
        reward = env_info.rewards[0]  # get the reward
        done = env_info.local_done[0]  # see if episode has finished
        episode_score += reward  # update the score

        if is_training:
            # Store transition <st,at,rt+1,st+1> in memory D
            agent.add_to_memory(state, action, reward, next_state, done)
            agent.replay()
        state = next_state  # roll over the state to next time step
        if done:  # exit loop if episode finished
            episode_rewards[episode] = episode_score
            last_m_rewards.append(episode_score)
            avg_score = np.mean(last_m_rewards)
            if avg_score > max_avg_score:
                max_avg_score = avg_score
            if episode % 100 == 0:
                print("\rEpisode {}/{} | Max Average Score: {} | Rolling average: {}".format(episode, total_episodes,
                                                                                             max_avg_score, avg_score),
                      end="")
                sys.stdout.flush()
            break

np.savetxt(f'results/rewards_{filename}.out', episode_rewards, delimiter=',')
if is_training:
    agent.checkpoint(f'./models/trained_agent_{filename}.pt')
env.close()
