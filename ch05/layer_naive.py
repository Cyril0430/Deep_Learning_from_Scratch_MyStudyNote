class MulLayer:
    def __init__(self): ##インスタンス変数（コンストラクタ）は、順伝播時の入力値を保持するために用いる。
        self.x = None
        self.y = None

    def forward(self, x, y):##乗算を実行する（順伝播での演算）
        self.x = x  # 第一引数をインスタンス変数（メンバ変数）に保存
        self.y = y  # 第二引数をインスタンス変数（メンバ変数）に保存
        out = x * y # 乗算

        return out
    
    def backward(self, dout):##逆伝播を計算する（実際に計算グラフを書いてみると分かりやすいだろう）
        dx = dout * self.y  # xとyをひっくり返す
        dy = dout * self.x

        return dx, dy

class AddLayer_ByMySelf:    # 乗算レイヤのクラス定義を参考にして自分で実装した「加算レイヤクラス」
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x  # 値の保存
        self.y = y
        out = x + y # 加算

        return out
    
    def backward(self, dout):
        dx = dout   # doutをそのまま後ろへ流す
        dy = dout

        return dx, dy
    

class AddLayer: ## 教科書の加算クラスの定義
    def __init__(self):
        pass        # `pass`は「何も行わない」という命令。つまり「インスタンス変数（コンストラクタ）は作らない」という意味（処理）になる

    def forward(self, x, y):
        out = x + y
        return out
    
    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1
        return dx, dy