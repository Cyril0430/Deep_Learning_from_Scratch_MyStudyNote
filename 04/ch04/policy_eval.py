# 反復方策評価の実装
import sys, os
sys.path.append(os.pardir)
from common.gridworld import GridWorld

def eval_onestep(pi, V, env, gamma=0.9):
    """反復方策評価のワンステップの更新のみを行う
    
    params:

    pi（defaultdict）: 方策
    V（defaultdict）: 価値関数
    env（GridWorld）: 環境
    gamma（float）: 割引率
    """
    for state in env.states():  # 1. 各状態へのアクセス
        if state == env.goal_state: # 2. ゴールの価値関数は常に0
            V[state] = 0    # エピソードタスクであるため、ゴールにおける価値関数の値は常に0になる。
            continue

        action_probs = pi[state]    # probsはprobabilitiesの略
        # pi <- defaultdict(lambda: {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25})
        new_V = 0   # 更新した価値関数を一時保存する（ここにはstateに格納されている状態における価値関数が一時保存される）。

        # 3. 各行動へアクセス
        for action, action_prob in action_probs.items():
            next_state = env.next_state(state, action)
            r = env.reward(state, action, next_state)
            # 4. 新しい価値関数（次の状態の価値関数から現在の状態の価値関数をもとめる）（下の更新式はベルマン方程式に他ならない）
            # V[next_state]が次の状態の価値関数である。
            new_V += action_prob * (r + gamma * V[next_state])
            # 最初の`new_V = 0`を消して`new_V = action_prob * (r + gamma * V[next_state])`とするのはダメなのか？
            # →`new_V = action_prob * (r + gamma * V[next_state])`としてしまうと、せっかく`for`文を回すことによって総和を表現しようとしているのに、それが実現されなくなってしまう。
            # つまり、for文と加算によって行動の候補に関する総和をとろうとしているのだ。
        V[state] = new_V
    
    return V

def policy_eval(pi, V, env, gamma, threshold=0.0001):
    while True:
        old_V = V.copy()    # 更新前の価値関数
        V = eval_onestep(pi, V, env, gamma)     # 価値関数の更新（反復方策評価ワンステップ分）

        # 更新された量の最大値を決める
        delta = 0
        for state in V.keys():
            t = abs(V[state] - old_V[state])    # `state`キーに格納されている価値関数の変化量を`t`としている。
            if delta < t:   # 更新された値（価値関数の変化量）が`delta`よりも大きい場合、
                delta = t   # `delta`に`t`の値を代入（更新）する。
        
        # 閾値との比較
        if delta < threshold:   # 閾値よりも`delta`（価値関数の変化量が最大であるもの）が小さい場合（超えない場合）、
            break               # 無限ループを抜け、処理を終える。
    
    return V