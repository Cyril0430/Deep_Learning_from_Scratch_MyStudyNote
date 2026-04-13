import os, sys
sys.path.append(os.pardir)
import numpy as np
from dezero import Model
from dezero import optimizers
import dezero.functions as F
import dezero.layers as L
from common.gridworld import GridWorld
import matplotlib.pyplot as plt

def one_hot(state):
    HEIGHT, WIDTH = 3, 4
    vec = np.zeros(HEIGHT * WIDTH, dtype=np.float32)
    y, x = state
    idx = WIDTH * y + x
    vec[idx] = 1.0
    return vec[np.newaxis, :]

class QNet(Model):
    def __init__(self):
        super().__init__()
        self.l1 = L.Linear(100) # hidden_size
        self.l2 = L.Linear(4)   # action_size
    
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x
    
class QLearningAgent:
    def __init__(self):
        self.gamma = 0.9
        self.lr = 0.01
        self.epsilon = 0.1
        self.action_size = 4

        self.qnet = QNet()
        self.optimizer = optimizers.SGD(self.lr)    # 最適化は確率勾配降下法で行う
        self.optimizer.setup(self.qnet)     # 適用するネットワークを指定
    
    def get_action(self, state):        # 注意：state は one-hot で入力する。
        if np.random.rand() > self.epsilon:
            return np.random.choice(self.action_size)
        else:
            qs = self.qnet(state)
            return qs.data.argmax()
    
    def update(self, state, action, reward, next_state, done):
        if done:
            next_q = np.zeros(1)    # [0.] ← こうすることで、後で、状態というデータ（第0次元目）の数が増えても、ブロードキャストによって計算可能になる。
        else:
            next_qs = self.qnet(next_state) # もちろん、next_state も one-hot ベクトル化されたものを想定している。next_qs の形状は (1, 4) 。
            next_q = next_qs.max(axis=1)    # 各状態において、行動を軸とする行動価値関数の最大値を next_q に代入。next_q の形状は (1,) 。
            next_q.unchain()    # ? → next_q は正解ラベルを作るために用いられる。教師あり学習では、正解ラベルに関する勾配は必要ないため、next_q.unchain() によってバックプロパゲーションの対象から除外している。
        
        target = self.gamma * next_q + reward
        qs = self.qnet(state)   # 今の状態（one-hot）における、行動毎の行動価値関数を qs に代入する。形状は (1, 4) 。
        q = qs[:, action]   # ? → 第0次元目は「状態」（ニューラルネットワークでは、これがデータ数に相当）で、第1次元目は、現在の行動。
        loss = F.mean_squared_error(target, q)

        self.qnet.cleargrads()
        loss.backward()
        self.optimizer.update() # オプティマイザを適用し、勾配によって重みパラメータとバイアスを更新。

        return loss.data    # アップデートした後の損失関数の値を出力する。

# ===== エージェントを動かす ===== #
# env = GridWorld()
# agent = QLearningAgent()

# episodes = 1000 # エピソード数
# loss_history = []

# for episode in range(episodes):
#     state = env.reset()
#     state = one_hot(state)
#     total_loss, cnt = 0, 0
#     done = False

#     while not done:
#         action = agent.get_action(state)
#         next_state, reward, done = env.step(action)
#         next_state = one_hot(next_state)

#         loss = agent.update(state, action, reward, next_state, done)
#         total_loss += loss
#         cnt += 1
#         state = next_state
    
#     average_loss = total_loss / cnt
#     loss_history.append(average_loss)