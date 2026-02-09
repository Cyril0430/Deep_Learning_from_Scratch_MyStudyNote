# L2正則化項を適用
import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.layer import *
from common.gradient import numerical_gradient
from collections import OrderedDict

class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01, weight_decay_lambda = 0.1):
        # 重みの初期化
        self.params = {}
        self.params["W1"] = weight_init_std * \
                            np.random.randn(input_size, hidden_size)
        self.params["b1"] = np.zeros(hidden_size)
        self.params["W2"] = weight_init_std * \
                            np.random.randn(hidden_size, output_size)
        self.params["b2"] = np.zeros(output_size)
        self.weight_decay_lambda = weight_decay_lambda
        
        # レイヤの生成
        self.layers = OrderedDict()         # 順序付きディクショナリ型の空データを用意
        # Python3.7以降では、通常のディクショナリ型にも順序が付いているが、今回の実装ではそれ以前のPythonでも実行できるような親切設計になっている。
        self.layers["Affine1"] = \
            Affine(self.params["W1"], self.params["b1"])
        self.layers["Relu1"] = Relu()
        self.layers["Affine2"] = \
            Affine(self.params["W2"], self.params["b2"])
        
        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():  # self.layersの番号付きディクショナリ型データから、格納されている値（各レイヤ（クラス）の参照データ）を走らせる。
            x = layer.forward(x)         # なるほど、layerには各処理のレイヤのインスタンスが保存されているから、そこでのforward処理（メソッド）を呼び出しているということね！
            # ああ、ここでもし、self.layers["Affine1"]やself.layers["Affine2"]に`myAffine`のインスタンスを代入していたら、わざわざ各要素（各レイヤ）を呼び出す形で書かなければならないということか！
        
        return x
    
    # x:入力データ, t:教師データ
    def loss(self, x, t):
        y = self.predict(x)
            #   + 0.5*self.weight_decay_lambda*(self.params["W1"]**2 + self.params["W2"]**2)  # 正則化項を加算
        loss_val = self.lastLayer.forward(y, t)
        weight_decay = 0
        weight_decay += 0.5*self.weight_decay_lambda * np.sum(self.params["W1"]**2)
        weight_decay += 0.5*self.weight_decay_lambda * np.sum(self.params["W2"]**2)
        return loss_val + weight_decay    # 「ソフトマックス関数からの損失関数の出力」のフォワード処理（メソッド）を呼び出しているのか。
        # もしここで、mySoftmaxWithLossをlatsLayerにインスタンス化していたら、TwoLayerNetのインスタンス変数を作るときに入力値を代入しなければならないのか。そうすると、TwoLayerNetの各メンバメソッド
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis = 1)      # 予測された結果（yのうち確率が最も高い要素番号をバッチ毎に出力）
        if t.ndim != 1:                 # 毎度疑問に思うが、もし「one-hotでない正解ラベルの行ベクトル」だとしたら、ここでバグが生じると思うのだが、これは問題ないのか？（あ、そもそももしバッチ数が１のときは、正解ラベルの教師データはint型をとるのか？）
                                        # いや、そもそも正解ラベル表現（One-hotでない）場合の形状は`(n,)`のようになるので、ここでの`if`文はスキップされるのか。
            t = np.argmax(t, axis = 1)  # 教師データが一次元でないとき（つまり、one-hot形式のとき）は、正解ラベルの行ベクトルに変換する（各バッチにおいて、1が格納されている要素番号を参照し、それらを行ベクトルとして順番に保存する）。
        
        accuracy = np.sum(y == t) / float(x.shape[0])   # 変数`accuracy`には、予測された結果と正解ラベルが一致している個数を、xの一次元目の数（つまり、バッチ数）で割ることで、単位試行あたりの正解数（正解率）が格納される。
        return accuracy
    
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)      # ラムダ式の引数として指定してある`W`はダミー変数か。そして`loss_W`には入力データと教師データによる損失関数の出力が返される

        grads = {}      # ディクショナリ型の箱（空データ）を用意
        grads["W1"] = numerical_gradient(loss_W, self.params["W1"])
        grads["b1"] = numerical_gradient(loss_W, self.params["b1"])
        grads["W2"] = numerical_gradient(loss_W, self.params["W2"])
        grads["b2"] = numerical_gradient(loss_W, self.params["b2"])

        return grads
    
    def gradient(self, x, t):       # ここが大本命の「誤差逆伝播法」の逆伝播処理の内容だ！
        # forward
        self.loss(x, t)     # まずは順伝播処理による誤差関数を出力する

        # backward
        dout = 1            # 損失関数の変化あたりの損失関数の変化率は1だからね
        dout = self.lastLayer.backward(dout)    # ここで最初の逆伝播処理を行う。インスタンス`lastLayer`だから、SoftmaxWithLossのbackward処理だね。

        layers = list(self.layers.values())     # self.layersに格納されているデータ（各レイヤの参照情報）のみを参照した番号付きディクショナリ型データをリスト型に変換する。
        layers.reverse()    # リスト型になった変数`layer`は今、順伝播のときの順番で値が並んでいるのでそれを逆の順番にする。そうすることで、逆伝播による処理の準備を行う。

        for layer in layers:
            dout = layer.backward(dout)         # ここで出力層以前へ向けた逆伝播処理が行われる（逆伝播の続きを行う）。
            # ちなみに、doutには入力値による勾配のみが格納されている。

        # Remark!：以降、`grads`というディクショナリ型のデータを作るが、そこには勾配の結果（パラメータを動かす方向）が格納されている。
        
        # 設定
        grads = {}
        grads["W1"] = self.layers["Affine1"].dW \
            + self.weight_decay_lambda*(self.params["W1"])    # ???
        grads["b1"] = self.layers["Affine1"].db     # あ！これはインスタンス`self.layers`のインスタンス変数`db`のことか！
        grads["W2"] = self.layers["Affine2"].dW \
            + self.weight_decay_lambda*(self.params["W2"])    # つまりここは、self.layersの「"Affine2"」のデータ、つまり「Affine(self.params["W2"], self.params["b2"])」というインスタンスが入っている（←これは正解 by AI）。そのインスタンスの「dW」というインスタンス変数（←これも正解 by AI）、つまり「self.params["W2"]」が`grads["W2"]`（←ここが間違い！ by AI)）というディクショナリ型データに格納される。

        # 間違っているところの修正：「self.params["W2"]」が`grads["W2"]`というディクショナリ型データに格納されるのではなく、Affineのインスタンス変数。TwoLayerNetではなく、Affineのインスタンス変数が格納される！
        grads["b2"] = self.layers["Affine2"].db     # ここは`self.layers["Affine2"].db`ではなく、`self.params["b2"]`にしてもよいのでは？
        # →だめ！今`self.params["b2"]`にはバイアスの初期値が格納されているため、せっかく勾配を計算したのに、初期値が代入されてしまう。一生更新されない。

        return grads