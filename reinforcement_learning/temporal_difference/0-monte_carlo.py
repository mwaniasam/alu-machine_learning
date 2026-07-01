#!/usr/bin/env python3
"""
Monte Carlo
"""
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1,
                gamma=0.99):
    """
    Performs the Monte Carlo algorithm
    """
    for ep in range(episodes):
        state = env.reset()
        episode = []
        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, reward))
            if done:
                break
            state = next_state

        G = 0
        for state, reward in reversed(episode):
            G = gamma * G + reward
            V[state] = V[state] + alpha * (G - V[state])
    return V
