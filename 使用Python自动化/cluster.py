import pandas as pd

data = pd.read_table('E:/课程/大数据/第二次作业/score.txt', encoding='utf-8')
a = [[i.split(',')[0], i.split(',')[1], i.split(',')[2]] for i in data['用户编号，商品编号，评分']]
df1 = pd.DataFrame(a)
df2 = df1.rename(columns={0: '用户编号', 1: '商品编号', 2: '评分'})
df4 = df2.set_index('用户编号')
uid = [int(i[0]) for i in df4.groupby('用户编号')]
uid.sort()

sid = [int(i[0]) for i in df4.groupby('商品编号')]
sid.sort()

data = pd.DataFrame(index=uid, columns=sid)
data

for row in df2.values:
    data.loc[int(row[0])][int(row[1])] = row[2]

data1 = data
data1 = data1.fillna(0)

# #数据集降维
from sklearn.decomposition import PCA

model_pca = PCA(n_components=2)  # n_components设置降维后的维度
X_pca = model_pca.fit(data1).transform(data1)

# 聚类
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

data = np.random.rand(100, 5)  # 生成一个随机数据，样本大小为100, 特征数为3

estimator = KMeans(n_clusters=5)  # 构造聚类器
y = estimator.fit_predict(X_pca)  # 聚类
label_pred = estimator.labels_  # 获取聚类标签
centroids = estimator.cluster_centers_  # 获取聚类中心

fig = plt.figure()
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, marker='*')
plt.scatter(centroids[:, 0], centroids[:, 1], marker='>')

plt.savefig("聚类结果.jpg")
plt.show()

group = {};
i = 0
for i in range(len(y)):
    if y[i] not in group:
        group[y[i]] = [uid[i]]
    else:
        group[y[i]].append(uid[i])

print(group)

groutT = {}
groutT[0] = list(range(1, 431))
groutT[1] = list(range(431, 556))
groutT[2] = list(range(556, 1001))
groutT[3] = list(range(556, 801))
groutT[4] = list(range(801, 1001))

print(groutT)

# In[25]:


import itertools

s = [list(item) for item in itertools.permutations([0, 1, 2, 3, 4], 5)]
# print(s)


# 计算精确率，召回率，F-measure
from sklearn.metrics import *

max = 0
trueS = []
for i in s:
    testS = []
    for j in uid:
        if j in groutT[0]:
            testS.append(i[0])
            continue
        if j in groutT[1]:
            testS.append(i[1])
            continue
        if j in groutT[2]:
            testS.append(i[2])
            continue
        if j in groutT[3]:
            testS.append(i[3])
            continue
        if j in groutT[4]:
            testS.append(i[4])
            continue
    if max < precision_score(testS, y, average='macro'):
        p = precision_score(testS, y, average='macro')
        r = recall_score(testS, y, average='macro')
        f1 = f1_score(testS, y, average='macro')
        trueS = i
print(trueS)
print(p, r, f1)

# In[ ]:
