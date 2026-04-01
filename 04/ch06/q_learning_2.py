from collections import defaultdict
import numpy as np
from common.utils import greedy_probs

# 下のものは分布モデル（方策を分布として保持し、それを用いて行動を選択している）
class preQLearningAgent:
    def __init__(self):
        self.gamma = 0.9
        self.alpha = 0.8
        self.epsilon = 0.1
        self.action_size = 4

        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        self.pi = defaultdict(lambda: random_actions)
        self.b = defaultdict(lambda: random_actions)
        self.Q = defaultdict(lambda: 0)
    
    def get_action(self, state):
        action_probs = self.b[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)
    
    def update(self, state, action, reward, next_state, done):
        if done:
            next_q_max = 0
        else:
            next_qs = [self.Q[next_state, a] for a in range(self.action_size)]
            next_q_max = max(next_qs)
        
        target = reward + self.gamma * next_q_max
        self.Q[state, action] += (target - self.Q[state, action]) * self.alpha

        # pi は greedy、b はε-greedy
        self.pi[state] = greedy_probs(self.Q, state, epsilon=0)
        self.b[state] = greedy_probs(self.Q, state, self.epsilon)

# サンプルモデル版のQ学習（つまり、方策を確率分布として保持しない。途中まで間違えて分布モデルで実装をしていたので、そこをコメントアウトしてある。）
class QLearningAgent:
    def __init__(self):
        self.gamma = 0.9
        self.alpha = 0.8
        self.epsilon = 0.1
        self.action_size = 4

        # random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        # self.pi = defaultdict(lambda: random_actions) # これは使わない
        # self.b = defaultdict(lambda: random_actions)  # 方策を確率分布として保持しないので、使わない。
        self.Q = defaultdict(lambda: 0)

    def get_action(self, state):
        # このタイミングでepsilon-greedy化
        # self.b[state] = greedy_probs(self.Q, state, self.epsilon) # しない。

        # action_probs = self.b[state]    # 挙動方策によって行動の確率変数を取得......しない。
        # actions = list(action_probs.keys())
        # probs = list(action_probs.values())
        # return np.random.choice(actions, p=probs)


        if np.random.rand() < self.epsilon:     # この時点でε-greedyに行動を選択する。つまりself.epsilon未満の場合、ランダムに行動を選び、
            return np.random.choice(self.action_size)
        else:                                   # それ以外の場合、Q関数を最大にする行動を選択する。
            qs = [self.Q[state, a] for a in range(self.action_size)]
            return np.argmax(qs)
    
    def update(self, state, action, reward, next_state, done):
        if done:
            next_q_max = 0
        else:
            next_qs = [self.Q[next_state, a] for a in range(self.action_size)]
            next_q_max = max(next_qs)
        
        target = self.gamma * next_q_max + reward
        self.Q[state, action] += (target - self.Q[state, action]) * self.alpha