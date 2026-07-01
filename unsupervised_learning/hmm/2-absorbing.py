#!/usr/bin/env python3
"""
Absorbing Chains
"""
import numpy as np


def absorbing(P):
    """
    Determines if a markov chain is absorbing
    """
    if not isinstance(P, np.ndarray) or len(P.shape) != 2:
        return False
    if P.shape[0] != P.shape[1]:
        return False

    n = P.shape[0]
    diag = np.diagonal(P)
    absorbing_states = np.where(diag == 1)[0]

    if len(absorbing_states) == 0:
        return False

    non_absorbing = set(range(n)) - set(absorbing_states)

    for i in non_absorbing:
        visited = set()
        queue = [i]
        can_reach = False
        while queue:
            curr = queue.pop(0)
            if curr in visited:
                continue
            visited.add(curr)
            if curr in absorbing_states:
                can_reach = True
                break
            for j in range(n):
                if P[curr, j] > 0 and j not in visited:
                    queue.append(j)
        if not can_reach:
            return False

    return True
