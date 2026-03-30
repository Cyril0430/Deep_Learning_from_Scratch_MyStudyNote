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