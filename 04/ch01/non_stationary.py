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