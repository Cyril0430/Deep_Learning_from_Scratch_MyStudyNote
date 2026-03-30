import numpy as np
from dezero import Variable
import dezero.functions as F

# トイ・データセット
np.random.seed(0)
x = np.random.rand(100, 1)  # 形状(100, 1)のランダムな要素を持つ行列が生成される
y = 5 + 2 * x + np.random.rand(100, 1)
x, y = Variable(x), Variable(y) # 省略可

W = Variable(np.zeros((1, 1)))
b = Variable(np.zeros(1))

def predict(x):
    y = F.matmul(x, W) + b
    return y

def mean_squared_error(x0, x1):
    diff = x0 - x1
    return F.sum(diff**2) / len(diff)

lr = 0.1
iters = 100

for i in range(iters):
    y_pred = predict(x)
    loss = mean_squared_error(y, y_pred)
    # もしくは loss = F.mean_squared_error(y, y_pred)
    W.cleargrad()
    b.cleargrad()
    loss.backward()

    W.data -= lr * W.grad.data
    b.data -= lr * b.grad.data

    if i % 10 == 0: # 10回ごとに出力
        print(loss.data)

print("===")
print(f"W = {W.data}")
print(f"b = {b.data}")