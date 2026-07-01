import numpy as np

def markov_chain(P, s, t=1):
    state = s
    for _ in range(t):
        state = np.matmul(state, P)
    return state

def regular(P):
    n = P.shape[0]
    if not np.allclose(np.sum(P, axis=1), 1):
        return None
    P_k = np.copy(P)
    is_reg = False
    for _ in range(n ** 2):
        if np.all(P_k > 0):
            is_reg = True
            break
        P_k = np.matmul(P_k, P)
    if not is_reg:
        return None
    evals, evecs = np.linalg.eig(P.T)
    idx = np.where(np.isclose(evals, 1))[0]
    if len(idx) == 0:
        return None
    steady_state = evecs[:, idx[0]].real
    steady_state = steady_state / np.sum(steady_state)
    return steady_state.reshape(1, n)

def absorbing(P):
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

print("testing markov chain")
P = np.array([[0.25, 0.2, 0.25, 0.3], [0.2, 0.3, 0.2, 0.3], [0.25, 0.25, 0.4, 0.1], [0.3, 0.3, 0.1, 0.3]])
s = np.array([[1, 0, 0, 0]])
print(markov_chain(P, s, 300))

print("testing regular")
a = np.eye(2)
b = np.array([[0.6, 0.4], [0.3, 0.7]])
print(regular(a))
print(regular(b))

print("testing absorbing")
print(absorbing(a))
print(absorbing(b))

