import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import *
import matplotlib
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读入数据
# score矩阵存放数据
score = np.zeros((1000, 200))
with open('E:/课程/大数据/第二次作业/score.txt', 'r') as f:
    for line in f.read().splitlines():
        i = int(line.split(',')[0]) - 1
        j = int(line.split(',')[1]) - 1
        k = float(line.split(',')[2])
        score[i][j] = k

# 拐点法判断k值
kmeans = [KMeans(n_clusters=k, random_state=23).fit(score) for k in range(1, 15)]
innertia = [model.inertia_ for model in kmeans]

plt.plot(range(1, 15), innertia)
plt.title('拐点法')
plt.xlabel('聚类数目')
plt.ylabel('内平方和')
plt.savefig("k值判断判断-拐点.jpg")
plt.show()
