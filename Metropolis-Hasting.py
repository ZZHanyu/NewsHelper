# Metropolis-Hasting Algorithm
import math
import random as rd
import time 
import numpy as np
import matplotlib.pyplot as plt



result_list = []

# First - 构建一个简单的指数分布采样：
# pi为目标分布
def function(x: int):
    pi = math.exp(-x)
    return pi





if __name__ == "__main__":
    # 生成当前值
    while len(result_list) < 10000:
        rd.seed(23312)
        x = rd.randint(1,1000)
        current_x = x
        # 生成提议分布
        # 这里使用curr value + 正态分布随机数
        proposed_x = x + np.random.normal(loc=0, size=1,scale=1)

        # 是否采纳？
        if (function(proposed_x) / function(current_x) > 1):
            result_list.append(proposed_x)
        else:
            result_list.append(current_x)

    for i in result_list:
        print(i)