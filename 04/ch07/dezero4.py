import numpy as np
from dezero import Model
from dezero import optimizers
import dezero.layers as L
import dezero.functions as F

# データセットの生成
np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

lr = 0.2
iters = 10000

class TwoLayerNet(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.l1 = L.Linear(hidden_size)
        self.l2 = L.Linear(out_size)
    
    def forward(self, x):
        y = F.sigmoid(self.l1(x))
        y = self.l2(y)
        return y
    
model = TwoLayerNet(10, 1)
optimizer = optimizers.SGD(lr)  # オプティマイザの生成（確率勾配降下法(Stochastic Gradient Descent)）、引き数に学習率を入れる。ここでいう確率(stochastic)とは、入力データからバッチ数の数だけランダムに取り出す様子を指している。
# optimizer = optimizers.Adam(lr)   # こうすることで、オプティマイザを Adam にすることができる。
optimizer.setup(model)  # オプティマイザにパラメータの更新を行わせるために、モデルを登録する。

for i in range(iters):
    y_pred = model(x)   # xから予測
    loss = F.mean_squared_error(y, y_pred)  # 予測結果と教師データ(yの値)の二乗平均誤差を計算

    model.cleargrads()
    loss.backward()

    optimizer.update()  # オプティマイザによる更新（backwardの後でいいの？）
                        # → param.data -= lr * param.grad.dataでやっていることと同じ
    if i % 1000 == 0:
        print(loss)