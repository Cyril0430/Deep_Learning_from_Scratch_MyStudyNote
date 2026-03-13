from collections import defaultdict
from policy_eval import *

def argmax(d):
    """ディクショナリに対して最大値を持つキーを返す
    
    param:

    d: ディクショナリ
    """
    max_value = max(d.values())
    max_key = 0
    for key, value in d.items():    # このような繰り返し処理のため、最大値が複数ある場合は、最後に位置する最大値を持つキーを返すようになる。
        if value == max_value:
            max_key = key
    
    return max_key

def greedy_policy(V, env, gamma):
    """価値関数をgreedy化（方策を更新する）メソッド
    
    params:

    V: 価値関数（初期値）
    env: GridWorldのインスタンス
    gamma: 割引率
    """
    pi = {} # 方策の箱を作る。ここに方策の更新履歴が残る。

    for state in env.states():  # 各状態を走らせる
        action_values = {}  # 行動価値関数

        for action in env.actions():    # その状態におけるすべての行動を走らせる
            next_state = env.next_state(state, action)  # 次の状態に遷移
            r = env.reward(state, action, next_state)   # 報酬関数から報酬を求める
            value = r + gamma * V[next_state]   # 次の状態における価値関数の値
            action_values[action] = value       # 次の状態における価値関数を保存
            max_action = argmax(action_values)  # 行動価値関数を最大にする行動（キー）を保存
            action_probs = {0: 0, 1: 0, 2: 0, 3: 0} # 行動をとる確率を初期化
            action_probs[max_action] = 1.0  # 行動価値関数を最大にする行動が生起する確率を1（決定論的）にする（max_actionが選ばれる確率が1になるように確率分布を生成する）
            pi[state] = action_probs    # 方策の箱に、行動価値関数を最大にする行動が生起する確率が1になるように設定された行動確率のディクショナリを、現在の状態毎に保存
    
    return pi

# 4.4.2 評価と改善を繰り返す（policy_iterメソッドの実装）
def policy_iter(env, gamma, threshold=0.001, is_render=False):
    """方策反復法による最適方策の導出メソッド
    
    params:

    env (Environment): 環境
    gamma (float): 割引率
    threshold (float): 方策評価を行うときの更新をストップするための閾値
    is_render (bool): 方策の評価・改善を行う過程を描画するかどうかのフラグ

    return 最適方策
    """
    pi = defaultdict(lambda: {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25})
    V = defaultdict(lambda: 0)

    while True:
        V = policy_eval(pi, V, env, gamma, threshold)   # 1. 評価（方策を求める）
        new_pi = greedy_policy(V, env, gamma)   # 2. 改善（求めた方策を改善する（教科書に則って言えば、方策をgreedy化する））

        if is_render:
            env.render_v(V, pi)
        
        if new_pi == pi:    # 3. 更新チェック（改善した方策がもとの方策と等しいときにwhileループを抜ける）
            break

        pi = new_pi     # 改善した方策がもとの方策と等しくないときは`pi`に更新した後の`new_pi`を保存し、次の繰り返しへ
    
    return pi