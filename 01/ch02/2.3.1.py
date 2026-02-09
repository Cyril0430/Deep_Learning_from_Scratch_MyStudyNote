def AND(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    tmp_theta_0 = w1*x1 + w2*x2
    if tmp_theta_0 <= theta:
        return 0
    elif tmp_theta_0 > theta:
        return 1

for x1_i in [0, 1]:
    for x2_i in [0, 1]:
        output = AND(x1_i, x2_i)
        print(output)