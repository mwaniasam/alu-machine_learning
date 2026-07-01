#!/usr/bin/env python3
"""
Play
"""
import numpy as np


def play(env, Q, max_steps=100):
    """
    Has the trained agent play an episode
    """
    state = env.reset()
    env.render()
    for _ in range(max_steps):
        action = np.argmax(Q[state, :])
        state, reward, done, _ = env.step(action)
        env.render()
        if done:
            break
    return reward
