import numpy as np
import os, sys
sys.path.append(os.pardir)
from common.layer import *
from common.util import *
from common.gradient import *
from common.function import *
from convolution_pooling_layer import *
from collections import OrderedDict

class SimpleConvNet:
    """簡単な畳み込みネット
    
    params: 

    input_dim: 入力データの（チャンネル, 高さ, 幅）の次元 \n
    conv_param: 畳み込み層のハイパーパラメータ（ディクショナリ）。ディクショナリキーは下記の通り。\n
        filter_num: フィルターの数 \n
        filter_size: フィルターのサイズ（一辺の長さ（要素数））\n
        pad: パディング \n
        stride: ストライド \n
    hidden_size: 隠れ層（全結合）のニューロン数 \n
    output_size: 出力層（全結合）のニューロン数 \n
    weight_init_std: 初期化の際の重みの標準偏差 \n

    """
    def __init__(self, input_dim=(1, 28, 28), conv_param={'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1}, hidden_size=100, output_size=10, weight_init_std=0.01):
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        input_size = input_dim[1]
        conv_output_size = \
            (input_size - filter_size + 2*filter_pad) / filter_stride + 1
        pool_output_size = \
            int(filter_num * (conv_output_size / 2) * (conv_output_size / 2))
            # conv_output_size / 2 : プーリングを行う領域の一辺の長さ（要素数）これは、プーリングサイズが暗黙的に2×2とされているので、畳み込み層からの出力画像の一辺を2で割ればプーリング層の出力画像の一辺の長さが出る。
            # pool_output_sizeでは最終的にプーリング層からの出力画像のサイズを計算している。
        
        self.params = {}
        # W1: 畳み込み層のフィルター
        self.params['W1'] = \
            weight_init_std * np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
        self.params['b1'] = np.zeros(filter_num)
        # W2: Affineレイヤの重みパラメータ
        self.params['W2'] = \
            weight_init_std * np.random.randn(pool_output_size, hidden_size)
        self.params['b2'] = np.zeros(hidden_size)
        self.params['W3'] = \
            weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b3'] = np.zeros(output_size)

        self.layers = OrderedDict()
        self.layers['Conv1'] = \
            Convolution(self.params['W1'], self.params['b1'], conv_param['stride'], conv_param['pad'])
        self.layers['Relu1'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers['Affine1'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W3'], self.params['b3'])

        self.last_layer = SoftmaxWithLoss()
    
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward()
        return x
    
    def loss(self, x, t):
        y = self.predict(x)
        return self.last_layer.forward(y, t)
    
    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 設定
        grads = {}
        grads['W1'] = self.layers['Conv1'].dW
        grads['b1'] = self.layers['Conv1'].db
        grads['W2'] = self.layers['Affine1'].dW
        grads['b2'] = self.layers['Affine1'].db
        grads['W3'] = self.layers['Affine2'].dW
        grads['b3'] = self.layers['Affine2'].db

        return grads