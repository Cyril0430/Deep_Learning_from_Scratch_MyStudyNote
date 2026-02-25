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
    
    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = int(1 + (H + 2*self.pad - FH) / self.stride)
        out_w = int(1 + (W + 2*self.pad - FW) / self.stride)

        col = im2col(x, FH, FW, self.stride, self.pad)  # `col`の形状は(N*out_h*out_w, C*FH*FW)
        col_W = self.W.reshape(FN, -1).T  # フィルターの展開
        out = np.dot(col, col_W) + self.b   # `(N*out_h*out_w, FN(=FH*FW))`

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

        return out