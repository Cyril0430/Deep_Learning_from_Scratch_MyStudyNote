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