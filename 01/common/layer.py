import numpy as np
from common.function import softmax, cross_entropy_error

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
        # ↑Affineレイヤをインスタンス化する際に重みパラメータとバイアスを指定するのは、外部（人の手）によって重みパラメータとバイアスを設定する必要がない（最初の設定を乱数によって規定すればよい）ためである。むしろ人の手によって両者のパラメータを変えるのはマズい（cf: カプセル化）。
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

# まずは自分で実装してみよう（何も見ずに）。
class mySoftmaxWithLoss:    
    def __init__(self, a):
        self.a = a          # 模範解答では保存していなかった。そもそも入力値a（模範解答ではx）は逆伝播の際に使用しないので、インスタンス変数として保存する必要が無いんだね。
        self.y = None
        self.t = None

    def forward(self, t):   # 教師データだけ入れる関数なんて、少し考えづらいんじゃない？「必要なデータを入れて初めて動く」っていう感覚があるから。
        self.y = softmax(self.a)
        self.t = t

        L = cross_entropy_error(self.y, self.t)
        return L
    
    def backward(self):     # どうせ損失関数の勾配は1だから、入れる必要がないと思い、インスタンス変数以外からの引数は指定しなかった。
        return self.y - self.t

# 模範解答
class SoftmaxWithLoss:      
    def __init__(self):
        self.loss = None    # 損失（なぜ、損失関数の値を保存しておくんだろう？）
        self.y = None       # softmaxの出力
        self.t = None       # 教師データ（one-hot vector）

    def forward(self, x, t):    # xが形状(N, m)のとき、
        self.t = t              # one-hotなので、tの形状も(N, m)である必要がある。
        self.y = softmax(x)     # self.yの形状は(N, m)になる。つまり各バッチに対し、正解である確率がm個格納されることになる。
        self.loss = cross_entropy_error(self.y, self.t) # 1次元のデータがself.lossに入る

        return self.loss
    
    def backward(self, dout = 1):   # なるほど、既定値を1にしておくのね。柔軟な対応が可能になるのか。
        batch_size = self.t.shape[0]        # バッチサイズを保存（コメントアウトの例だとbatch_sizeは「N」になる。
        dx = (self.y - self.t) / batch_size # なんでバッチサイズで割るの？
                                            # →ここで用いた交差エントロピー誤差が「誤差の総和をバッチサイズの分だけ割る」ことで定義されていたため、`self.y - self.t`は「誤差の総和」になってしまう。そこでその誤差の総和をバッチサイズで割ることで、順伝播だけでなく逆伝播においても総和の誤差勾配の平均を伝えることになり、数学的に等価になる。

        return dx

# なんでバッチサイズで割るのか？という疑問について
# 数理的な理由は、ノートに書いた。
# 2. 学習の安定性（直感的な理由）
# もしバッチサイズで割らなかった場合、**「バッチサイズを変えると学習率も変えなければならなくなる」**という不便なことになります。

# 例えば、バッチサイズが10個の時と、100個の時を比べてみましょう。

# 割らない場合（総和）:

# バッチ10個の勾配の強さ： 約10個分

# バッチ100個の勾配の強さ： 約100個分（10倍になってしまう！）

# → データ量が増えると勾配が巨大になり、パラメータが一気に更新されすぎて学習が発散（オーバーシュート）してしまいます。→学習率を変える必要がある。

# 割る場合（平均）：

# バッチ10個でも100個でも、「データ1個あたりの勾配の強さ」に正規化されます。

# → バッチサイズを変えても、勾配の大きさの尺度が変わらないため、同じ学習率（Learning Rate）を使って安定して学習できます。