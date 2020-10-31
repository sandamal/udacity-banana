[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 1: Navigation
![Trained Agent][image1]
### Introduction

In this project, we train an agent to navigate (and collect bananas!) in a large, square world.  

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic. When the agent gets an average score of +13 over 100 consecutive episodes, the environment is considered to be solved.

### Getting Started

1. Follow the instructions in this [link](https://github.com/udacity/deep-reinforcement-learning#dependencies) to install the required dependencies.

2. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

3. Place the file in project folder, and unzip (or decompress) the file. Make sure the following line in the **`banana_world.py`** points to the correct location.
```python
env = UnityEnvironment(file_name="./Banana_Linux/Banana.x86_64")
```
### Instructions

The project has the following main files:
- Memory.py - stores the agent's experiences.
- QNetwork.py - the neural network model using pyTorch.
- DQNetwork.py - the DQN using instances of Memory and QNetwork for training and predicting.
- banana_world.py - the main script.

You can run the **`banana_world.py`** script in training or testing mode. To train the agent, make sure the **`is_training`** variable is set to **`True`** in the **`banana_world.py`** file.  
By default, the code is configured to run in testing mode using saved weights.

