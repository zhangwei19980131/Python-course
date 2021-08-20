import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from matplotlib import pyplot
from sklearn.datasets._samples_generator import make_blobs

np.set_printoptions(threshold=np.inf)
# 读取文本数据
# score矩阵存放数据
score = np.zeros((1000, 200))
with open('E:/课程/大数据/第二次作业/score.txt', 'r') as f:
    for line in f.read().splitlines():
        i = int(line.split(',')[0]) - 1
        j = int(line.split(',')[1]) - 1
        k = float(line.split(',')[2])
        score[i][j] = k
print(type(score))
