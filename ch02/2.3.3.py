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

for x1_i in [0, 1]:
    for x2_i in [0, 1]:
        print(f"連言の否定について、x1={x1_i}、x2={x2_i}のとき、{NAND(x1_i, x2_i)}となる。")

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

for x1_i in [0, 1]:
    for x2_i in [0, 1]:
        print(f"選言について、x1={x1_i}、x2={x2_i}のとき、{OR(x1_i, x2_i)}となる。")