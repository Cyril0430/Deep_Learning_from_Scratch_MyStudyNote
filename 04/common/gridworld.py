import numpy as np
import common.gridworld_render as render_helper

class GridWorld:
    """3×4マスのグリッドワールド
    
    インスタンス
    """
    def __init__(self):
        self.action_space = [0, 1, 2, 3]    # 行動の候補（その詳細は`self.action_meaning`にて定義している。
        self.action_meaning = {     # 行動の候補に対応した行動の形態の詳細
            0: "UP",
            1: "DOWN",
            2: "LEFT",
            3: "RIGHT",
        }

        self.reward_map = np.array( # 報酬マップ（マップの各マスに対する報酬。リンゴがあるところは`1.0`で爆弾があるところには`-1.0`が報酬として設定されている。
            [[0, 0, 0, 1.0],
             [0, None, 0, -1.0],
             [0, 0, 0, 0]]
        )
        self.goal_state = (0, 3)    # ゴールの位置（ここに位置するとタスクが終了し、初期位置に戻る）
        self.wall_state = (1, 1)    # 入れない壁が存在する位置
        self.start_state = (2, 0)   # エージェントの初期位置
        self.agent_state = self.start_state # エージェントの現在の状態（初期値はスタート位置に存在している）
    
    @property   # 直後のメソッドをインスタンス変数として扱うことができる！
    def height(self):
        return len(self.reward_map) # self.reward_mapの0次元目の長さ
    
    @property
    def width(self):
        return len(self.reward_map[0])  # self.reward_mapの`[0, :]`成分（0行目）の長さ
    
    @property
    def shape(self):
        return self.reward_map.shape    # self.reward_mapの形状（`(3, 4)`）
    
    def actions(self):   # 全ての行動にアクセス
        return self.action_space    # [0, 1, 2, 3]
    
    def states(self):   # 全ての状態にアクセス（一回の処理毎に出力する）
        for h in range(self.height):    # ここで、メソッドを`self.height`というインスタンス変数として使用している！
            for w in range(self.width): # ここでも！
                yield (h, w)            # yieldを用いることで、一回の実行毎にfor文で得られる結果を返すことができる。

    # 環境の状態遷移を表すメソッドと報酬関数のメソッドを実装
    
    def next_state(self, state, action):    # 環境の状態遷移を表すメソッド
        # 1. 移動先の場所の計算
        action_move_map = [(-1, 0), (1, 0), (0, -1), (0, 1)]    # 場所の変化量（左から「DOWN」「UP」「LEFT」「RIGHT」。数学の座標とはxとyが逆なので注意！）
        move = action_move_map[action]  # どちらに動くか決定（左右上下のいずれか）
        next_state = (state[0] + move[0], state[1] + move[1])   # 次の状態に遷移する（y軸, x軸）
        ny, nx = next_state # 次の状態のx座標（nx）とy座標（ny）が設定される。

        # 2. 移動先がグリッドワールドの枠の外か、それとも移動先が壁か？
        if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:   # 次の状態が枠外のとき
            next_state = state  # 状態をもとに戻す
        elif next_state == self.wall_state: # 次の状態が壁のとき
            next_state = state  # 状態をもとに戻す
        
        # 3. 次の状態を返す
        return next_state   # 今回は状態遷移が決定論的なので、そのまま（確率1で）次の状態を返す。
    
    def reward(self, state, action, next_state):    # 報酬関数を表すメソッド
        return self.reward_map[next_state]  # この本では報酬関数を決定論的なものとみなしているので、報酬をそのまま返す。しかし今回は状態遷移が決定論的であるため、次の状態のみによって報酬を決定している点に注意。
    
    def reset(self):    # エージェントの位置をリセットするメソッド
        self.agent_state = self.start_state
        return self.agent_state
    
    def step(self, action): # エージェントの行動によって時間を一つ進めるようなメソッド
        state = self.agent_state
        next_state = self.next_state(state, action)
        reward = self.reward(state, action, next_state)
        done = (next_state == self.goal_state)

        self.agent_state = next_state
        return next_state, reward, done
    
    # 次の関数はグリッドを可視化するものであり、メソッドの定義の中身はあまり重要ではない
    def render_v(self, v=None, policy=None, print_value=True):
        renderer = render_helper.Renderer(self.reward_map, self.goal_state,
                                          self.wall_state)
        renderer.render_v(v, policy, print_value)