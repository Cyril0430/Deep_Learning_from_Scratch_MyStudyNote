import numpy as np

def softmax(x):
    """softmax関数
    
    param:

    x: NumPy配列
    """
    x = x - np.max(x, axis=-1, keepdims=True)   # オーバーフロー対策
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

def cross_entropy_error(y, t):  # これ、yの形状が(m,)でtの形状が(m,)のone-hotでない教師データのときに不具合を起こすのでは？→この場合、どちらも(1, m)にreshapeされる。すると`t.size == y.size`が成り立つので、`t`は「正解ラベルの内一番大きいもの」が代入される。形状が(1,)になってしまう。そうすると、教師データの「正解ラベルと出力結果が各要素で対応している」という機能が失われてしまう。
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    
    # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
    if t.size == y.size:        # y（予測の確率値）とt（教師データ）のデータの形状が等しいとき（つまり、tがone-hotのとき）
        t = t.argmax(axis = 1)  # これで本当に正解ラベルのインデックスに変換できるの？
                                # →はい、できます。教師データの各バッチに格納されている要素のうち一番大きい値、つまり1が格納されている要素番号が`t.argmax(axis = 1)`によって計算されるので、それが改めて`t`に代入される。つまり生成されるのは正解ラベルのインデックスが格納された行ベクトルである。
    
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_grad(x):##誤差逆伝播法のために用いるもの（シグモイド関数と同じ（式変形したもの）だが、プログラムの処理の観点からだと異なる可能性がある。いずれにせよ、チャプター4の時点では吟味の必要はない。
    return (1.0 - sigmoid(x)) * sigmoid(x)