#Set seed to be 505 for the reproducible results

import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.
        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.alpha = 0.02
        self.gamma = 0.99
        self.seed = np.random.seed(505)

    def epsilon_greedy_probs(self, state, i_episode):
        self.epsilon = 1.0 / i_episode
        policy = np.ones(self.nA) * self.epsilon / self.nA
        policy[np.argmax(self.Q[state])] = 1 - self.epsilon + (self.epsilon / self.nA)
        # policy[np.argmax(self.Q[state])] += 1 - self.epsilon
        return policy

    def select_action(self, state, i_episode):
        """ Given the state, select an action.
        Params
        ======
        - state: the current state of the environment
        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        policy = self.epsilon_greedy_probs(state, i_episode)
        return np.random.choice(self.nA, p=policy)

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.
        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """

        if not done:
            self.Q[state][action] += self.alpha * (reward + self.gamma * np.max(self.Q[next_state]) - self.Q[state][action])

        else:
            self.Q[state][action] += self.alpha * (reward - self.Q[state][action])
