import sys, os
sys.path.append(os.pardir)
import numpy as np

# 自作レイヤ（まずは教科書だけを見て、お手本コードは見ずに実装）
class myBatchNormalization:
    def __init__(self, gamma = 1, beta = 0):
        self.gamma = gamma
        self.beta = beta
        self.x = None
        self.mean = None
        self.variance = None
        self.norm = None

    def forward(self, x):
        self.x = x
        self.mean = np.mean(x, axis = 0)
        self.variance = np.mean((x - self.mean)**2, axis = 0)
        self.norm = (x - self.mean) / np.sqrt(self.variance + 10e-7)
        out = self.gamma * self.norm + self.beta

        return out
    
    def backward(self, d_out):
        d_mean = 1 / self.x.shape[0]
        d_variance = (2 / d_mean) * (self.x - self.mean)
        d_norm = \
            (2*(1 - d_mean)*(self.variance + 10e-7) - (self.x - self.mean)*d_variance) / (2*(self.variance + 10e-7)*np.sqrt(self.variance + 10e-7))

        d_gamma = np.sum(d_out*self.norm, axis = 0)
        d_beta = np.sum(d_out, axis = 0)
        dx = np.sum(d_out*self.gamma*d_norm, axis = 0)

        return d_gamma, d_beta, dx

# AIが出力したお手本コード
class BatchNormalization:
    def __init__(self, gamma = 1, beta = 0):
        self.gamma = gamma
        self.beta = beta
        self.x = None
        self.mean = None
        self.variance = None
        self.norm = None
    
    def forward(self, x):
        self.x = x
        self.mean = np.mean(x, axis = 0)
        self.variance = np.mean((x - self.mean)**2, axis = 0)
        self.norm = (x - self.mean) / np.sqrt(self.variance + 10e-7)
        out = self.gamma * self.norm + self.beta

        return out

    def backward(self, dout):   # ここだけが違った
        # 準備
        N, D = self.x.shape
        mu = self.mean
        var = self.variance
        eps = 10e-7
        std_inv = 1. / np.sqrt(var + eps)
        x_mu = self.x - mu

        # パラメータの勾配（ここまでは合っている）
        d_gamma = np.sum(dout * self.norm, axis = 0)
        d_beta = np.sum(dout, axis = 0)

        # 【Step 1】正規化された値（norm）への勾配
        d_norm = dout * self.gamma

        # 【Step 2】分散（variance）への勾配
        # 全データの勾配を合計する（axis = 0）
        d_variance = np.sum(d_norm * x_mu, axis = 0) * -0.5 * (std_inv**3)

        # 【Step 3】平均（mean）への勾配
        # 平均は「直接の影響」と「分散を通じた影響」の2つのルートがある
        d_mean = np.sum(d_norm, axis = 0) * std_inv + d_variance * np.sum(x_mu, axis = 0) / (-2 / N)    # 結局、「分散から通じた影響」は`0`になるから、足さなくてもよい。

        # 【Step 4】入力（x）への勾配
        # xは「norm」「variance」「mean」のすべてに影響を与えている
        dx = d_norm * std_inv + (d_mean / N) + (d_variance * 2 + x_mu / N)

        return d_gamma, d_beta, dx