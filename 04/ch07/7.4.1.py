import numpy as np

# 状態（位置）というカテゴリデータをone-hot表現に変換する関数を実装する。
def one_hot(state):
    HEIGHT, WIDTH = 3, 4    # 状態の縦幅・横幅
    vec = np.zeros(HEIGHT * WIDTH, dtype=np.float32)    # 形状(12,)の、float32型のデータが格納されているNumPy配列を生成
    y, x = state
    idx = WIDTH * y + x     # ? → stateの座標`(x, y)`をone_hot表現にしたときに、どこが1になるのかを指定するためのインデックス。x が3まで行き、4になろうとした瞬間 y に 1 繰り上がるイメージ。x が 4 で繰り上がり、（存在しないが）y は 3 で繰り上がる。そのような進数表記を10進数に直す処理をここでは行っている。
    vec[idx] = 1.0          # ? → idxのところを1にする。
    return vec[np.newaxis, :]   # バッチのための新しい軸を追加。np.newaxis を[]によるインデックスの中で使うと、その位置にサイズが 1 の新たな次元が追加される。

state = (2, 0)
x = one_hot(state)  # 0次元目が「状態の種類」、1次元目が「その状態のone_hot表現」となるような設計かな？

print(x.shape)  # (1, 12)
print(x)    # [[0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. ]]