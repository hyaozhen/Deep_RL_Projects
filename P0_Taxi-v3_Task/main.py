#monitor.py matches the orignal algorithm. Epsilon decays by episodes rather than by time steps within each episode.
#Set seed to be 505 for the reproducible results

import gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D

#Sarsa
from agent_Sarsa import Agent
from monitor_Sarsa import interact
env = gym.make('Taxi-v3')
env.seed(505)
agent = Agent()
avg_rewards_1, best_avg_reward = interact(env, agent)

#Sarsamax
from agent_Sarsamax import Agent
from monitor_Sarsamax import interact
env = gym.make('Taxi-v3')
env.seed(505)
agent = Agent()
avg_rewards_2, best_avg_reward = interact(env, agent)

#ExpectedSarsa
from agent_ExpectedSarsa import Agent
from monitor_ExpectedSarsa import interact
env = gym.make('Taxi-v3')
env.seed(505)
agent = Agent()
avg_rewards_3, best_avg_reward = interact(env, agent)

#Plot
line1, = plt.plot(np.linspace(0,20000,len(avg_rewards_1),endpoint=False),np.asarray(avg_rewards_1),label='Sarsa')
line2, = plt.plot(np.linspace(0,20000,len(avg_rewards_2),endpoint=False),np.asarray(avg_rewards_2),label='Sarsamax(Q-Learning)')
line3, = plt.plot(np.linspace(0,20000,len(avg_rewards_3),endpoint=False),np.asarray(avg_rewards_3),label='Expected Sarsa')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=4)})
plt.title('Comparison of Performance')
plt.xlabel('Episode Number')
plt.ylabel('Average Reward (Over Next %d Episodes)' % 100)
plt.show()
