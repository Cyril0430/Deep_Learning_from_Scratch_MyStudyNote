# 自作レイヤ
import sys, os
sys.path.append(os.pardir)
import numpy as np

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
    
    def backward(self, d_out):
        d_mean = 1 / self.x.shape[0]
        d_variance = (2 / d_mean) * (self.x - self.mean)
        d_norm = \
            (2*(1 - d_mean)*(self.variance + 10e-7) - (self.x - self.mean)*d_variance) / (2*(self.variance + 10e-7)*np.sqrt(self.variance + 10e-7))

        d_gamma = np.sum(d_out*self.norm, axis = 0)
        d_beta = np.sum(d_out, axis = 0)
        dx = np.sum(d_out*self.gamma*d_norm, axis = 0)

        return d_gamma, d_beta, dx