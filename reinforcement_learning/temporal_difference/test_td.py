import gym
import numpy as np

def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99):
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

def td_lambtha(env, V, policy, lambtha, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99):
    for ep in range(episodes):
        state = env.reset()
        E = np.zeros_like(V)
        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            
            # Should we use V[next_state] if done?
            delta = reward + gamma * V[next_state] - V[state]
            E[state] += 1.0
            V += alpha * delta * E
            E *= gamma * lambtha
            
            if done:
                break
            state = next_state
    return V

def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    initial_epsilon = epsilon
    for ep in range(episodes):
        state = env.reset()
        E = np.zeros_like(Q)
        
        p = np.random.uniform()
        if p < epsilon:
            action = np.random.randint(env.action_space.n)
        else:
            action = np.argmax(Q[state])
            
        for _ in range(max_steps):
            next_state, reward, done, _ = env.step(action)
            
            p = np.random.uniform()
            if p < epsilon:
                next_action = np.random.randint(env.action_space.n)
            else:
                next_action = np.argmax(Q[next_state])
                
            delta = reward + gamma * Q[next_state, next_action] - Q[state, action]
            E[state, action] += 1.0
            Q += alpha * delta * E
            E *= gamma * lambtha
            
            if done:
                break
            state = next_state
            action = next_action
            
        # Update epsilon
        epsilon = min_epsilon + (initial_epsilon - min_epsilon) * np.exp(-epsilon_decay * ep)
    return Q

# Test MC
np.random.seed(0)
env = gym.make('FrozenLake8x8-v0')
env.seed(0)
LEFT, DOWN, RIGHT, UP = 0, 1, 2, 3
def policy(s):
    p = np.random.uniform()
    if p > 0.5:
        if s % 8 != 7 and env.desc[s // 8, s % 8 + 1] != b'H': return RIGHT
        elif s // 8 != 7 and env.desc[s // 8 + 1, s % 8] != b'H': return DOWN
        elif s // 8 != 0 and env.desc[s // 8 - 1, s % 8] != b'H': return UP
        else: return LEFT
    else:
        if s // 8 != 7 and env.desc[s // 8 + 1, s % 8] != b'H': return DOWN
        elif s % 8 != 7 and env.desc[s // 8, s % 8 + 1] != b'H': return RIGHT
        elif s % 8 != 0 and env.desc[s // 8, s % 8 - 1] != b'H': return LEFT
        else: return UP

V = np.where(env.desc == b'H', -1, 1).reshape(64).astype('float64') 
np.set_printoptions(precision=4)
print("MC:")
print(monte_carlo(env, V.copy(), policy, episodes=1000, max_steps=50, alpha=0.05, gamma=0.9).reshape((8, 8)))

# Test TD
np.random.seed(0)
env.seed(0)
print("TD:")
print(td_lambtha(env, V.copy(), policy, 0.8, episodes=500, max_steps=25, alpha=0.07, gamma=0.95).reshape((8, 8)))

# Test Sarsa
np.random.seed(0)
env.seed(0)
Q = np.random.uniform(size=(64, 4))
print("Sarsa:")
print(sarsa_lambtha(env, Q.copy(), 0.8, episodes=2000, max_steps=60, alpha=0.04, gamma=0.91, epsilon=0.95, min_epsilon=0.05, epsilon_decay=0.025)[:5])
