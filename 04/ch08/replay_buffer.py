from collections import deque
import random
import numpy as np

# 経験再生の実装を行う
class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
    
    def add(self, state, action, reward, next_state, done):
        """バッファにデータを追加する"""
        data = (state, action, reward, next_state, done)
        self.buffer.append(data)
    
    def __len__(self):      # バッファのサイズ（長さ）を返す
        return len(self.buffer)
    
    def get_batch(self):
        data = random.sample(self.buffer, self.batch_size)  # self.buffer から self.batch_size の数だけランダムに選び、それを data に格納する。

        state = np.stack([x[0] for x in data])  # state の第0次元目に沿って data から取得した状態に関するデータを積み重ねていく。実は np.array によっても取得することが可能。
        action = np.array([x[1] for x in data]) # data に格納されている各データの1次元目にある行動データを取得
        reward = np.array([x[2] for x in data])
        next_state = np.stack([x[3] for x in data])
        done = np.array([x[4] for x in data]).astype(int)
        return state, action, reward, next_state, done

import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")
replay_buffer = ReplayBuffer(buffer_size=10000, batch_size=32)

for episode in range(10):
    state, info = env.reset()
    done = False

    while not done:
        # action = 0    # action を 0 、つまり左向きへと常に行動する
        if random.random() < 0.5:
            action = 0
        else:
            action = 1
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        replay_buffer.add(state, action, reward, next_state, done)
        state = next_state

state, action, reward, next_state, done = replay_buffer.get_batch()
print("\n=== バッファーの各形状 ===\n")
print(f"state の形状: {state.shape}")   # (32, 4)
print(f"action の形状: {action.shape}") # (32,)
print(f"reward の形状: {reward.shape}") # (32,)
print(f"next_state の形状: {next_state.shape}") # (32, 4)
print(f"done の形状: {done.shape}\n") # (32,)

true_count = sum([x[4] for x in replay_buffer.buffer])
print(f"実施した done のうち、True の数: {true_count}")