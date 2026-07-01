#!/usr/bin/env python3
"""
SARSA(λ)
"""
import numpy as np


def epsilon_greedy(state, Q, epsilon, env):
    """
    Epsilon greedy policy
    """
    p = np.random.uniform()
    if p < epsilon:
        action = np.random.randint(env.action_space.n)
    else:
        action = np.argmax(Q[state])
    return action


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1,
                  gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Performs SARSA(λ)
    """
    initial_epsilon = epsilon
    for ep in range(episodes):
        state = env.reset()
        E = np.zeros_like(Q)

        action = epsilon_greedy(state, Q, epsilon, env)

        for _ in range(max_steps):
            next_state, reward, done, _ = env.step(action)
            next_action = epsilon_greedy(next_state, Q, epsilon, env)

            if done:
                delta = reward - Q[state, action]
            else:
                delta = reward + gamma * Q[next_state, next_action] - \
                    Q[state, action]

            E[state, action] += 1.0
            Q = Q + alpha * delta * E
            E = E * gamma * lambtha

            if done:
                break
            state = next_state
            action = next_action

        # Update epsilon
        epsilon = min_epsilon + (initial_epsilon - min_epsilon) * \
            np.exp(-epsilon_decay * ep)
    return Q
