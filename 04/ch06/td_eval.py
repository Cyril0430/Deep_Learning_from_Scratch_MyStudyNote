from collections import defaultdict
import numpy as np

class TdAgent:
    def __init__(self):
        self.gamma = 0.9
        self.alpha = 0.01
        self.action_size = 4

        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        self.pi = defaultdict(lambda: random_actions)
        self.V = defaultdict(lambda: 0)
    
    def get_action(self, state):
        action_probs = self.pi[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)
    
    def eval(self, state, reward, next_state, done):
        next_V = 0 if done else self.V[next_state]  # ゴールの価値関数は0（価値関数は将来得られる報酬の割引率付きの総和である。ゴールにおける価値関数は、その先には何も無いので、常に0になる。
        target = reward + self.gamma * next_V

        self.V[state] += (target - self.V[state]) * self.alpha