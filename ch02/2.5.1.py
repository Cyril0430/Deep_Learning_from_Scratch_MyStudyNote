# XOR（排他的論理和）のパーセプトロンの実装について
# これまでのように単一のパーセプトロンでは実装できないので、XOR, OR, ANDを組み合わせた多層パーセプトロンを採用する。
# これは同値変形のようなもの（つまり、排他的論理演算子で結合された命題変数と同値である命題変数を、XORとORとANDを用いて変形すること）
import numpy as np
def XOR(x1, x2):
    # NANDのパーセプトロン計算
    x = np.array([x1, x2])
    w_nand = np.array([-0.5, -0.5])
    b_nand = 0.7
    tmp_nand = np.sum(w_nand*x) + b_nand
    # NANDのパーセプトロン計算の結果をs1に保存する
    if tmp_nand <= 0:
        s1 = 0
    else:
        s1 = 1

    # ORのパーセプトロン計算
    w_or = np.array([0.5, 0.5])
    b_or = -0.25
    tmp_or = np.sum(w_or*x) + b_or
    # ORのパーセプトロン計算の結果をs2に保存する
    if tmp_or <= 0:
        s2 = 0
    else:
        s2 = 1
    
    # ANDのパーセプトロンを、s1とs2という再帰的入力から計算する
    # -> XORのパーセプトロン（多層パーセプトロン）を計算することになる
    s = np.array([s1, s2])
    w_and = np.array([0.5, 0.5])
    b_and = -0.7
    tmp_and = np.sum(w_and*s) + b_and
    if tmp_and <= 0:
        return 0
    else:
        return 1

for x1_i in [0, 1]:
    for x2_i in [0, 1]:
        print(
            f"排他的論理和について、x1 = {x1_i}、x2 = {x2_i}のとき、{XOR(x1_i, x2_i)}となる。"
        )