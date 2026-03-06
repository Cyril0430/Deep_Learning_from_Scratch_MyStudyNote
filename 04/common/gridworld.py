import numpy as np

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
    
    def action(self):
        return self.action_space    # [0, 1, 2, 3]
    
    def states(self):
        for h in range(self.height):    # ここで、メソッドを`self.height`というインスタンス変数として使用している！
            for w in range(self.width): # ここでも！
                yield (h, w)