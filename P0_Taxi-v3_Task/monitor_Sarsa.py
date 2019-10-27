from collections import deque
import sys
import math
import numpy as np
import matplotlib.pyplot as plt

def interact(env, agent, num_episodes=20000, window=100):
    """ Monitor agent's performance.

    Params
    ======
    - env: instance of OpenAI Gym's Taxi-v1 environment
    - agent: instance of class Agent (see Agent.py for details)
    - num_episodes: number of episodes of agent-environment interaction
    - window: number of episodes to consider when calculating average rewards

    Returns
    =======
    - avg_rewards: deque containing average rewards
    - best_avg_reward: largest value in the avg_rewards deque
    """
    # initialize average rewards
    avg_rewards = deque(maxlen=num_episodes)
    # initialize best average reward
    best_avg_reward = -math.inf
    # initialize monitor for most recent rewards
    samp_rewards = deque(maxlen=window)
    # for each episode
    for i_episode in range(1, num_episodes+1):
        # begin the episode, set s0
        state = env.reset()
        # set a0
        action = agent.select_action(state, i_episode)
        # initialize the sampled reward
        samp_reward = 0
        while True:
            # agent performs the selected action
            next_state, reward, done, _ = env.step(action)
            # update the sampled reward
            samp_reward += reward
            # agent performs internal updates based on sampled experience
            next_action = agent.step(state, action, reward, next_state, done, i_episode)
            # update the state (s <- s') to next time step
            state = next_state
            # update the state (a <- a') to next time step
            action = next_action
            if done:
                # save final sampled reward
                samp_rewards.append(samp_reward)
                break
        if (i_episode >= 100):
            # get average reward from last 100 episodes
            avg_reward = np.mean(samp_rewards)
            # append to deque
            avg_rewards.append(avg_reward)
            # update best average reward
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
        # monitor progress
        print("\rEpisode {}/{} || Best average reward {}".format(i_episode, num_episodes, best_avg_reward), end="")
        sys.stdout.flush()
        # check if task is solved (according to OpenAI Gym)
        if best_avg_reward >= 9.7:
            print('\nEnvironment solved in {} episodes.'.format(i_episode), end="")
            break
        if i_episode == num_episodes: print('\n')
    # plot performance
    plt.plot(np.linspace(0,num_episodes,len(avg_rewards),endpoint=False),np.asarray(avg_rewards))
    plt.title('Performance of Sarsa')
    plt.xlabel('Episode Number')
    plt.ylabel('Average Reward (Over Next %d Episodes)' % window)
    plt.show()
    return avg_rewards, best_avg_reward
