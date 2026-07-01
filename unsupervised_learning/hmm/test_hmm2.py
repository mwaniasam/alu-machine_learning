import numpy as np

def forward(Observation, Emission, Transition, Initial):
    T = Observation.shape[0]
    N, M = Emission.shape
    F = np.zeros((N, T))
    F[:, 0] = Initial[:, 0] * Emission[:, Observation[0]]
    for t in range(1, T):
        F[:, t] = np.matmul(F[:, t-1], Transition) * Emission[:, Observation[t]]
    P = np.sum(F[:, T - 1])
    return P, F

def viterbi(Observation, Emission, Transition, Initial):
    T = Observation.shape[0]
    N, M = Emission.shape
    V = np.zeros((N, T))
    B = np.zeros((N, T), dtype=int)
    V[:, 0] = Initial[:, 0] * Emission[:, Observation[0]]
    B[:, 0] = 0
    for t in range(1, T):
        for j in range(N):
            trans_probs = V[:, t-1] * Transition[:, j]
            V[j, t] = np.max(trans_probs) * Emission[j, Observation[t]]
            B[j, t] = np.argmax(trans_probs)
    P = np.max(V[:, T-1])
    best_last_state = np.argmax(V[:, T-1])
    path = [best_last_state]
    for t in range(T-1, 0, -1):
        best_last_state = B[best_last_state, t]
        path.append(best_last_state)
    path.reverse()
    return path, P

def backward(Observation, Emission, Transition, Initial):
    T = Observation.shape[0]
    N, M = Emission.shape
    B = np.zeros((N, T))
    B[:, T-1] = 1
    for t in range(T-2, -1, -1):
        term = Emission[:, Observation[t+1]] * B[:, t+1]
        B[:, t] = np.matmul(Transition, term)
    P = np.sum(Initial[:, 0] * Emission[:, Observation[0]] * B[:, 0])
    return P, B

def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    T = Observations.shape[0]
    M, N_out = Emission.shape
    for _ in range(iterations):
        F = np.zeros((M, T))
        F[:, 0] = Initial[:, 0] * Emission[:, Observations[0]]
        for t in range(1, T):
            F[:, t] = np.matmul(F[:, t-1], Transition) * Emission[:, Observations[t]]
        B = np.zeros((M, T))
        B[:, T-1] = 1
        for t in range(T-2, -1, -1):
            B[:, t] = np.matmul(Transition, Emission[:, Observations[t+1]] * B[:, t+1])
        xi = np.zeros((M, M, T - 1))
        for t in range(T - 1):
            denominator = np.sum(F[:, t] * np.matmul(Transition, Emission[:, Observations[t+1]] * B[:, t+1]))
            if denominator == 0:
                return None, None
            for i in range(M):
                xi[i, :, t] = F[i, t] * Transition[i, :] * Emission[:, Observations[t+1]] * B[:, t+1] / denominator
        gamma = np.zeros((M, T))
        for t in range(T - 1):
            gamma[:, t] = np.sum(xi[:, :, t], axis=1)
        gamma[:, T-1] = F[:, T-1] * B[:, T-1]
        gamma_sum = np.sum(gamma[:, T-1])
        if gamma_sum == 0:
            return None, None
        gamma[:, T-1] /= gamma_sum
        Transition_new = np.sum(xi, axis=2) / np.sum(gamma[:, :-1], axis=1).reshape((M, 1))
        Emission_new = np.zeros((M, N_out))
        for k in range(N_out):
            Emission_new[:, k] = np.sum(gamma[:, Observations == k], axis=1) / np.sum(gamma, axis=1)
        if np.allclose(Transition, Transition_new) and np.allclose(Emission, Emission_new):
            return Transition_new, Emission_new
        Transition = Transition_new
        Emission = Emission_new
    return Transition, Emission

np.random.seed(1)
Emission = np.array([[0.90, 0.10, 0.00], [0.40, 0.50, 0.10]])
Transition = np.array([[0.60, 0.4], [0.30, 0.70]])
Initial = np.array([0.5, 0.5])
Hidden = [np.random.choice(2, p=Initial)]
for _ in range(364):
    Hidden.append(np.random.choice(2, p=Transition[Hidden[-1]]))
Observations = [np.random.choice(3, p=Emission[s]) for s in Hidden]
Observations = np.array(Observations)
T_test = np.ones((2, 2)) / 2
E_test = np.abs(np.random.randn(2, 3))
E_test = E_test / np.sum(E_test, axis=1).reshape((-1, 1))
T_res, E_res = baum_welch(Observations, T_test, E_test, Initial.reshape((-1, 1)))
print("T_res:")
print(np.round(T_res, 2))
print("E_res:")
print(np.round(E_res, 2))

