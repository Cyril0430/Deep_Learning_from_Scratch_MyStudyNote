import numpy as np
# import gym
import gymnasium

# env = gym.make("CartPole-v1", render_mode="rgb_array")
env = gymnasium.make("CartPole-v1", render_mode="rgb_array")
state = env.reset()
done = False

while not done:
    env.render()
    action = np.random.choice([0, 1])   # 0 か 1 かをランダムに選ぶ
    next_state, reward, terminated, truncated, info = env.step(action)  # 教科書では 4 つの返り値を想定していた。
    done = terminated or truncated  # 教科書にはない部分。教科書に合わせるために追記。
env.close()