# 先ほど作成したXOR, OR, ANDのパーセプトロンをここにコピペしておく
import numpy as np

def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7    # 「バイアス」はニューロンの発火のしやすさを表す（まぁ、閾値を変換しただけだから、その通りだよな）。
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1
    
# 連言の否定(NAND)のパーセプトロン
def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

# 選言(OR)のパーセプトロン
def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.25
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

# 排他的論理和を既に作成した関数（メソッド）で実装
def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y

for x1_i in [0, 1]:
    for x2_i in [0, 1]:
        print(f"排他的論理和について、x1={x1_i}、x2={x2_i}のとき、{XOR(x1_i, x2_i)}となる。")