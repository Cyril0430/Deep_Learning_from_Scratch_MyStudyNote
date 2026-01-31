import numpy as np

def softmax(x):
    """softmax関数
    
    param:

    x: NumPy配列
    """
    x = x - np.max(x, axis=-1, keepdims=True)   # オーバーフロー対策
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    
    # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
    if t.size == y.size:
        t = t.argmax(axis = 1)  # これで本当に正解ラベルのインデックスに変換できるの？
    
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_grad(x):##誤差逆伝播法のために用いるもの（シグモイド関数と同じ（式変形したもの）だが、プログラムの処理の観点からだと異なる可能性がある。いずれにせよ、チャプター4の時点では吟味の必要はない。
    return (1.0 - sigmoid(x)) * sigmoid(x)