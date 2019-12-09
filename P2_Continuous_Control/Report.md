# Project 2. continuous Control

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

## 0. Overviews

### DDPG Algorithm

I implemented DDPG algorithm and I will explain the details of DDPG in the section 5. DDPG of this report.

![ddpg.png](https://miro.medium.com/max/1084/1*BVST6rlxL2csw3vxpeBS8Q.png)

### The Hyperparameters Used
Both Actor and Critic have a learning rate of 1e-4. In order to get the best strategy, I gradually reduced the probability of exploration. The mean value of noise processing is 0, and the sigma is 0.2. The Actor network is a three-layer neural network whose first layer has 256 units, the second layer has 128 units, and the final output layer has 4 units. The Critic network is constructed using a three-layer neural network similar to the number of units in the Actor network, except that the actions are connected to the output of the first layer and a single output unit of the last layer. For faster learning, both networks use the elu activation function. Replay buffer size is 1e5, minibatch size is 128, discount factor is 0.99, soft update of target parameters TAU is 1e-3, learning rate of the actor and critic is 1e-4, and L2 weight decay is 0. Exploration Decay Rate begins from 1.0 with decay rate 0.05 to 3e-5.

### Result
![Q3IIN6.png](https://s2.ax1x.com/2019/12/05/Q3IIN6.png)

The agents solved the enviroment and achieved +30 rewards in 70 episodes.

### Future Improvements

Methods to further improve agent performance:
  - Use Proximal Policy Optimization algorithm.
  - Use Prioritized Experience Replay

## 1. learning objectives
 - Understand the difference between value-based method and policy-based method;

 - Understand the objective function and optimization idea of the Policy Gradient method;

- Understand monte carlo strategy gradient and Actor-Critic method;


## 2. Policy-based Methods
Some reinforcement Learning algorithms learned above, such as q-learning, SARSA, and DQN, are value-based algorithms, that is, to get a Q table or approximate a value function, and then formulate strategies based on the learned Q table or Q function. This article will introduce the algorithm is another idea, direct learning strategy function.

Value based method such as DQN, input is state, output is Q value, with the state-action mapping to Q value, when making a decision in a certain state, select the action with the maximum Q value, this is the strategy of DQN. But here are a few problems:

Value based method such as DQN, input is state, output is Q value, with the state-action mapping to Q value, when making a decision in a certain state, select the action with the maximum Q value, this is the strategy of DQN. But here are a few problems:

- An arbitrarily small change in the estimated value function may cause the corresponding action to be selected or not to be selected. Such discontinuous change is an important factor that causes the method based on the value function to fail to obtain the convergence guarantee.

- Selecting the maximum Q value is very difficult in high latitude or continuous space;

- No random strategy can be learned, and in some cases random strategy is often the optimal strategy.

## 3. Policy Gradient Methods

![499A2825-C572-489F-9D49-577C3EAD6929.png](https://i.loli.net/2019/12/05/aSg1EVpODRkcwGZ.png)

### Problem Setup
![8A386693-1D63-486D-BD60-460AFE06ACCF.png](https://i.loli.net/2019/12/05/NOnAaZL7VoyEv2q.png)
- *Trajectory- = state-action sequence
- *Trajectory- is a fancy way of referring episode, but it does *not- track *rewards*
- *Trajectory- has *no- limit on its *length*. So, it can correspond to a full episode or just a small part of an episode.

![Trajectory T_ state-action sequence.png](https://i.loli.net/2019/12/05/LrnEYqglQN9BjyS.png)
We denote the length with a capital H, where H stands for Horizon.
We denote a Trajectory with the Greek letter Tau.
Then, the sum reward from that Trajectory is written as R of Tau.

### What are Policy Gradient Methods?
- *Policy-based methods- are a class of algorithms that search directly for the optimal policy, without simultaneously maintaining value function estimates.
- *Policy gradient methods- are a subclass of policy-based methods that estimate the weights of an optimal policy through gradient ascent.
- In this lesson, we represent the policy with a neural network, where our goal is to find the weights θ of the network that maximize expected return.

### The Big Picture
- The policy gradient method will iteratively amend the policy network weights to:
	- make (state, action) pairs that resulted in positive return more likely, and
	- make (state, action) pairs that resulted in negative return less likely.

### REINFORCE

![EF2CCC89-11BA-4CB4-8C64-31F646793686.png](https://i.loli.net/2019/12/05/2b7p3h5vIeZzY4Q.png)

## 4. Actor-Critic Methods
![E4A9CEF8-6194-4269-9B56-7C4CAA83E6B3.png](https://i.loli.net/2019/12/05/AgDd8bor7pewBlZ.png)

If we train a neural network to approximate a value function and then use it as a baseline, would this make for a better baseline, and if so, would a better baseline further reduce the variance of policy-based methods?

Indeed. In fact, that’s basically all *actor-critic methods- are trying to do, *to use value-based techniques to further reduce the variance of policy-based methods*.

![EED30642-1744-4094-A2C1-FFB68821F3F3.png](https://i.loli.net/2019/12/05/y6jOSfGZK47hX2q.png)

- A big part of the effort in reinforcement learning and research is an attempt to reduce the variance of algorithms while keeping bias to a minimum.
- You know by now that a reinforcement learning agent tries to find policies to maximize the total expected reward.
- But since we’re limited to sampling the environment, we can only estimate these expectation.
- The question is, what’s the best way to estimate value functions for our actor-critic methods.

## 5. DDPG

In the  [DDPG paper](https://arxiv.org/abs/1509.02971) , they introduced this algorithm as an “Actor-Critic” method. Though, some researchers think DDPG is best classified as a DQN method for continuous action spaces (along with  [NAF](https://arxiv.org/abs/1603.00748) ). Regardless, DDPG is a very successful method and it’s good for you to gain some intuition.

![9FBBF671-0ECC-4849-82F6-6D085BF79B2D.png](https://i.loli.net/2019/12/05/uwfB1QLEzAT79GI.png)
One of the limitations of the DQN agent is that it
is not straightforward to use in continuous action spaces.
Imagine a DQN network that takes inner state and outputs the action value function.
For example, for two action, say, up and down, Q(s, “up”) gives you the estimated expected value for selecting the up action in state s, say minus 2.18.
Q(s “down”), gives you the estimated expected value for selecting the down action in state s, say 8.45.
To find the max action value function for this state, you just calculate the max of these values. Pretty easy.
It’s very easy to do a max operation in this example because this is a discrete action space.
Even if you had more actions say a left, a right, a jump and so on,
you still have a discrete action space.
Even if it was high dimensional with many, many more actions, it would still be doable.

![8AC821D2-C7E6-432F-AE34-686805B8905F.png](https://i.loli.net/2019/12/05/NKOeuQkSd9FicoA.png)
In DDPG, we use two deep neural networks.
We can call one the actor and the other the critic.
Nothing new to this point.
Now, the actor here is used to approximate the optimal policy deterministically.
That means we want to always output the best believed action for any given state.
This is unlike a stochastic policies in which we want the policy to learn a probability distribution over the actions.

![B6BC761E-FEFB-4184-B171-4DBE42E1F192.png](https://i.loli.net/2019/12/05/kNjVUQbhLiMqdyn.png)
In DDPG, we want the believed best action every single time we query the actor network.
That is a deterministic policy.
The actor is basically learning the argmax a Q(S, a), which is the best action.

![Q3IEtK.md.png](https://s2.ax1x.com/2019/12/05/Q3IEtK.md.png)
The critic learns to evaluate the optimal action value function by using the actors best believed action.

![Q3IVfO.md.png](https://s2.ax1x.com/2019/12/05/Q3IVfO.md.png)
Again, we use this actor, which is an approximate maximizer, to calculate a new target value for training the action value function, much in the way DQN does.
