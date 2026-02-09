import numpy as np

class NonStatBandit:
    def __init__(self, arms = 10):
        self.arms = arms
        self.rates = np.random.rand(arms)   # 標準正規分布から乱数を生成
    
    def play(self, arm):
        rate = self.rates[arm]
        self.rates += 0.1 * np.random.rand(self.arms)   # ノイズ追加
        if rate > np.random.rand():
            return 1
        else:
            return 0

class AlphaAgent:
    def __init__(self, epsilon, alpha, actions = 10):
        self.epsilon = epsilon
        self.Qs = np.zeros(actions)
        self.alpha = alpha
    
    def update(self, action, reward):
        # alpha で更新
        self.Qs[action] += (reward - self.Qs[action]) * self.alpha
        # self.Qs[action] += (reward - self.Qs[action]) / self.ns[action]   # 割る数が実施回数に依存している

    def get_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, len(self.Qs))
        return np.argmax(self.Qs)