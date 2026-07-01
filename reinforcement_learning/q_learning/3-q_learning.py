#!/usr/bin/env python3
"""
Q-learning
"""
import numpy as np
epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99,
          epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Performs Q-learning
    """
    total_rewards = []
    initial_epsilon = epsilon
    
    for ep in range(episodes):
        state = env.reset()
        episode_reward = 0

        for _ in range(max_steps):
            action = epsilon_greedy(Q, state, epsilon)
            next_state, reward, done, _ = env.step(action)

            # Keep original reward for tracking
            track_reward = reward

            if done and reward == 0:
                reward = -1

            delta = reward + gamma * np.max(Q[next_state, :]) - \
                Q[state, action]
            Q[state, action] = Q[state, action] + alpha * delta

            episode_reward += track_reward

            if done:
                break
            state = next_state

        total_rewards.append(episode_reward)
        epsilon = min_epsilon + (initial_epsilon - min_epsilon) * \
            np.exp(-epsilon_decay * ep)

    return Q, total_rewards
