import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.util import *

# まずは順伝播のみ
class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

        # 中間データ(backward時に使用)
        self.x = None
        self.col = None
        self.col_W = None

        # 重み・バイアスパラメータの勾配
        self.dW = None
        self.db = None
    
    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = int(1 + (H + 2*self.pad - FH) / self.stride)
        out_w = int(1 + (W + 2*self.pad - FW) / self.stride)

        col = im2col(x, FH, FW, self.stride, self.pad)  # `col`の形状は(N*out_h*out_w, C*FH*FW)
        col_W = self.W.reshape(FN, -1).T  # フィルターの展開
        out = np.dot(col, col_W) + self.b   # `(N*out_h*out_w, FN)`

        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)
        # out.reshape(N, out_h, out_w, -1)までは、次のような処理がされているイメージ↓

        # post_out[0, 0, 0, 0] <- pre_out[0, 0]
        # post_out[0, 0, 0, 1] <- pre_out[0, 1]
        # ...
        # post_out[0, 0, 0, FN-1] <- pre_out[0, FN-1]
        # post_out[0, 0, 1, 0] <- pre_out[1, 0]
        # ...
        # post_out[N-1, out_h-1, out_w-1, FN-1] <- pre_out[N*out_h*out_w-1, FN-1]

        # `.transpose(0, 3, 1, 2)`をすると、outの並びは`(N, FN, out_h, out_w)`となる。

        self.x = x
        self.col = col  # (N*out_h*out_w, C*FH*FW)
        self.col_W = col_W  # (C*FH*FW, FN)

        return out

    def backward(self, dout):
        """
        dout: 上流から伝わってきた勾配（N, FN, out_h, out_w）
        """
        FN, C, FH, FW = self.W.shape

        # 1. doutの形状を2次元配列に戻す（順伝播の最後の`reshape`&`transpose`の逆再生）
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)

        # 2. バイアス`b`の勾配
        self.db = np.sum(dout, axis=0)

        # 3. 重み`W`の勾配（im2colの出力（col）の転置と上流から伝わってきた勾配を`(N*out_h*out_w, FN)`の形状にしたものの行列積で求める）
        self.dW = np.dot(self.col.T, dout)  # (C*FH*FW, N*out_h*out_w) dot (N*out_h*out_w, FN) -> (C*FH*FW, FN)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        # 4. 入力 x のパッチ空間での勾配（dcol）
        dcol = np.dot(dout, self.col_W.T)

        # 5. dcolを元の画像空間 dx に復元する（ここで col2im が用いられる）
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx

class Pooling:
    def __init__(self, pool_h, pool_w, stride=2, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        # 展開 (1)

        # ↓このコードの吟味から（処理の流れを思い浮かべよう）
        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)    # col: (N*padded_out_h*padded_out_w, C*self.pool_h*self.pool_w)
        col = col.reshape(-1, self.pool_h*self.pool_w)  # (N*padded_out_h*padded_out_w*C, self.pool_h*self.pool_w)