import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import *
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
score = pd.DataFrame(score)
kmeans = [KMeans(n_clusters=k).fit(score) for k in range(1, 15)]

# 轮廓系数指标
from sklearn.metrics import silhouette_score

silhouette_scores = [silhouette_score(score, model.labels_) for model in kmeans[1:10]]

plt.figure(figsize=(8, 3))
plt.plot(range(1, 10), silhouette_scores, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("轮廓系数", fontsize=14)

plt.savefig("k值判断-轮廓系数.jpg")
plt.show()
