### Deep Reinforcement Learning - Collaboration and Competition

#### William Galindez Arias


##### Table of Contents

1. Introduction
2. Learning Algorithm 
3. Hyper-parameters
4. Experimentation
5. Implementation and plot of rewards


#### 1. Introduction

Tennis is environment used for this project. Here, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The agents will need to compete and learn how to cooperate to keep the game going on. 

#### 2. Learning Algorithm 

The chosen algorithm was the Deep Deterministic Policy Agent DDPG with few modifications tailored to the Tennis environment. This algorithm was chosen given that: 
- It is a general purpose algorithm that converges to learned policies that exploit local information or agent observations at execution time.
- Can be applied to cooperative, competitive and mixed interactions
In addition to the main reasons exposed above, it was also considered:
- Essentially consist of an actor-critic, model free algorithm with policy gradient
- Implements a *Replay Buffer*. Learning in mini-batches rather than online
- 
#### 2.1 Changes applied to the DDPG for this project

- Taking as a starting point the algorithms and techniques used in the second project, and given the fact that the loss of the learning algorithm is calculated from **experiences sampled from a replay buffer**
the buffer sized was decreased and adjust, finding out that a smaller *replay buffer* helped in the convergence of the algorithm to average scores higher than 0.5 in less time

- Having less agents, compared to the previous project where the policy had to be learned for 20 agents, in this case 2 agents are present in the environment. The *actor* 
- and *critic* were implemented with two fully connected layers with 256 neurons instead of 400 as in the previous continous control project
- The scores and average scored calculated were modified to get the max score between the two scores yielded by the environment after the two agents have played

#### 3. Hyper-parameters & Network architecture

**Network Architecture **

**Actor**:
 ```    self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

 ```
 
 **Critic**
  ``` 
 self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.fc2 = nn.Linear(fcs1_units+action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        
  ``` 
  
 Observation Space: 
 ```
  There are 2 agents. 
  Each observes a state with length: 24
The state for the first agent looks like:
[ 0.          0.          0.          0.          0.          0.    
  0.  0.          0.          0.          0.          0.          0.      
  0.  0.          0.         -7.38993645 -1.5        -0.          0.
  6.83172083  5.99607611 -0.          0.        ]

```

the general layer or networks architecture consist of fully connected layers with ReLU activation function, as can be also seen in `model.py`file in this repository

**Hyper-parameters**

``` 
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-2              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-4        # learning rate of the critic
WEIGHT_DECAY = 0.000    # L2 weight decay
UPDATE_EVERY = 1        # how often to update the networks
``` 
