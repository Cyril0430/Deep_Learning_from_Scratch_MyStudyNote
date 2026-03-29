import sys, os
sys.path.append(os.pardir)

import numpy as np
from collections import defaultdict, deque
from common.utils import greedy_probs

# 教科書に書かれたクラスを参考にして作成（継承・オーバーライドしやすいように変更）
class SarsaOffPolicyAgent:
    def __init__(self):
        self.gamma = 0.9
        self.alpha = 0.8
        self.epsilon = 0.1
        self.action_size = 4
        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        self.pi = defaultdict(lambda: random_actions)
        self.b = defaultdict(lambda: random_actions)
        self.Q = defaultdict(lambda: 0)
        self.memory = deque(maxlen=2)
    
    def get_action(self, state):
        action_probs = self.b[state]    # 1. 挙動方策から取得
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)

    def reset(self):
        self.memory.clear()

    def _calculate_target(self, reward, rho, next_q):
        return rho * (reward + self.gamma * next_q)

    def update(self, state, action, reward, done):
        self.memory.append((state, action, reward, done))
        if len(self.memory) < 2:
            return
        
        state, action, reward, done = self.memory[0]
        next_state, next_action, _, _ = self.memory[1]

        if done:
            next_q = 0  # ゴールの次の状態における行動価値は0
            rho = 1 # ゴールの次の状態におけるTDターゲットには、重み付けによる補正の必要が無いので、rho=1
        else:
            next_q = self.Q[next_state, next_action]
            # 2. 重みrhoを求める（self.pi[a][b]とすると、self.piというディクショナリのaというキーの値（ディクショナリ型）のbというキーの値を参照する）
            rho = self.pi[next_state][next_action] / self.b[next_state][next_action]
        
        # 3. rhoによるTDターゲットの補正
        # target = rho * (reward + self.gamma * next_q)
        target = self._calculate_target(reward, rho, next_q)
        self.Q[state, action] += (target - self.Q[state, action]) * self.alpha
        # 4. それぞれの方策を改善
        self.pi[state] = greedy_probs(self.Q, state, 0) # piの方は完全なgreedyなのでepsilonの値は0にする。
        self.b[state] = greedy_probs(self.Q, state, self.epsilon)

# ノイズの少ない、教科書の改善版
class NeoSarsaOffPolicyAgent(SarsaOffPolicyAgent):
    def __init__(self):
        super().__init__()
    
    def _calculate_target(self, reward, rho, next_q):
        return reward + rho * self.gamma * next_q