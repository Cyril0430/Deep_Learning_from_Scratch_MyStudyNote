import sys, os
sys.path.append(os.pardir)

from collections import defaultdict, deque
import numpy as np
from common.utils import greedy_probs

class SarsaAgent:
    def __init__(self):
        self.gamma = 0.9
        self.alpha = 0.8
        self.epsilon = 0.1
        self.action_size = 4

        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        self.pi = defaultdict(lambda: random_actions)
        self.Q = defaultdict(lambda: 0)
        self.memory = deque(maxlen=2)   # 1. dequeを使う
    
    def get_action(self, state):
        action_probs = self.pi[state]   # 2. piから選ぶ
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)
    
    def reset(self):
        self.memory.clear()

    def update(self, state, action, reward, done):
        self.memory.append((state, action, reward, done))   # 状態、行動、報酬、ゴールしたか否かの情報をタプルとして保存（二つまで）
        if len(self.memory) < 2:
            return  # 関数処理を中断する
        
        state, action, reward, done = self.memory[0]
        next_state, next_action, _, _ = self.memory[1]
        # 3. 次のQ関数
        next_q = 0 if done else self.Q[next_state, next_action] # $Q_\pi(S_{t+1}, A_{t+1})$の計算

        # 4. TD法による更新
        target = reward + self.gamma * next_q   # TDターゲットの定義
        self.Q[state, action] += (target - self.Q[state, action]) * self.alpha

        # 5. 方策の改善
        self.pi[state] = greedy_probs(self.Q, state, self.epsilon)