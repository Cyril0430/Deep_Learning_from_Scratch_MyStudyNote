from collections import deque
import random
import numpy as np

# 経験再生の実装を行う
class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
    
    def add(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        self.buffer.append(data)
    
    def __len__(self):      # バッファのサイズ（長さ）を返す
        return len(self.buffer)
    
    def get_batch(self):
        data = random.sample(self.buffer, self.batch_size)  # self.buffer から self.batch_size の数だけランダムに選び、それを data に格納する。

        state = np.stack([x[0] for x in data])  # state の第0次元目に沿って data から取得した状態に関するデータを積み重ねていく。そもそも状態は numpy 配列として定義されているので、それをさらに ndarray 型データのリストで囲むことはできない。だから stack を用いているのだ。
        action = np.array([x[1] for x in data]) # data に格納されている各データの1次元目にある行動データを取得
        reward = np.array([x[2] for x in data])
        next_state = np.stack([x[3] for x in data])
        done = np.array([x[4] for x in data]).astype(int)
        return state, action, reward, next_state, done