#!/usr/bin/env python3
"""
TD(λ)
"""
import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000, max_steps=100,
               alpha=0.1, gamma=0.99):
    """
    Performs the TD(λ) algorithm
    """
    for ep in range(episodes):
        state = env.reset()
        E = np.zeros_like(V)
        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, done, _ = env.step(action)

            delta = reward + gamma * V[next_state] - V[state]
            E[state] += 1.0
            V = V + alpha * delta * E
            E = E * gamma * lambtha

            if done:
                break
            state = next_state
    return V
