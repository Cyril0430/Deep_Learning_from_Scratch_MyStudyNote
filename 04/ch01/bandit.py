import numpy as np

class Bandit:
    def __init__(self, arms=10):
        self.rates = np.random.rand(arms)   # 各マシンの確率
    
    def play(self, arm):
        rate = self.rates[arm]
        if rate > np.random.rand():
            return 1
        else:
            return 0

class Agent:
    def __init__(self, epsilon, action_size = 10):
        """Agentクラス
        
        params:\n

        epsilon: ${\epsilon}$-greedy法における、ランダムな行動をとる確率

        action_size: エージェントが選択できる行動の数
        """
        self.epsilon = epsilon
        self.Qs = np.zeros(action_size)
        self.ns = np.zeros(action_size)
    
    def update(self, action, reward):
        """スロットマシンの価値を推定する
        
        params:\n
        
        action: 選択したスロットマシンの番号

        reward: 報酬
        """
        self.ns[action] += 1
        self.Qs[action] += (reward - self.Qs[action]) / self.ns[action]

    def get_action(self):
        """探索か活用化を${\epsilon}$をもとに決定する
        
        no params
        """
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, len(self.Qs))
        
        return np.argmax(self.Qs)