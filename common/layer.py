import numpy as np

class Relu:
    def __init__(self):
        self.mask = None    # 入力情報を保存するための箱
    
    def forward(self, x):
        self.mask = (x <= 0)    # x配列において、0以下の要素のみをTrueとする。つまりself.maskはbool型のリストが代入される
        out = x.copy()          # copyメソッド：`5.5.1.ipynb`にて解説したので、そこを参照せよ。
        out[self.mask] = 0          # self.maskのboolリストのTrueの位置が0になるようにしてoutを書き換える。

        return out
    
    def backward(self, dout):
        # if self.mask:     # このような条件分岐だと、dxが配列型ではなく整数型のみになってしまうのでマズい。
        #     dx = 0
        # else:
        #     dx = dout

        dout[self.mask] = 0
        dx = dout           # このように、dout配列の、`self.mask`がTrueの部分、つまり入力される配列における0以下の部分だけを0にするということだ。そうしてから、dxにdoutを代入すれば、「x > 0のときはそのまま後ろに流し、x <= 0のときは0を流す」ということが表現できる！
        
        return dx

class Sigmoid:
    def __init__(self):
        # self.y = None
        self.out = None     # プログラミング実装では、変数名から動作の意図や意味が明らかである方が好まれるのでこちらにしましょう。
    
    def forward(self, x):
        # self.mask = x
        out = 1 / (1 + np.exp(-x))
        self.out = out

        return out
    
    def backward(self, dout):       # 逆伝播の際に「順伝播時に保存されると都合が良い変数」をインスタンス変数として定義しておくとよい！そして、それをもとにして順伝播では保存する動作も含め、逆伝播時は保存した変数を用いて実装する、という流れが適切かもしれない！（私の考察）
        dx = dout * self.out * (1.0 - self.out)

        return dx

        # return dout * self.out * (1.0 -  self.out)

class myAffine:     # 自分で作ってみた
    def __init__(self):
        self.W = None
        self.x = None
        self.b = None

    def forward(self, W, x, b):     # これだと、forwardメンバメソッドを呼び出すたびにWとbを渡さなければならない。それだと重みとバイアスの管理が大変（模範解答のforwardメソッドを参照せよ）。
        self.W = W
        self.x = x
        self.b = b

        y = np.dot(self.x, self.W) + self.b

        return y
    
    def backward(self, dout):
        dx = np.dot(dout, np.transpose(self.W))
        dW = np.dot(np.transpose(self.x), dout)
        db = np.sum(dout, axis = 0)

        return dx, dW, db
    
class Affine:       # 教科書の模範解答（四次元のデータも考慮した実装になっている）
    def __init__(self, W, b):   # インスタンス変数を外部から定義できるようにしているのか。なぜか？
        # なるほど、Wとbはクラスの外部で学習するためか…？
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None
    
    def forward(self, x):       # これなら、わざわざforward()を実行するたびにWとbを記入する必要がない。
        self.x = x
        out = np.dot(x, self.W) + self.b

        return out
    
    def backward(self, dout):
        dx = np.dot(dout, self.W.T)     # self.W.Tとnp.transpose(self.W)の違いって、何だろう？
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis = 0)

        return dx   # ほう、dxのみ出力するのか、そうか、xだけがデータに直接関連するからか。Wやbの変化を扱う意義は無いからか。