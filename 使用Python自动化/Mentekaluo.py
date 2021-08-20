from scipy.stats import norm
import random
import matplotlib.pyplot as plt


# 定义平稳分布
def smooth_dist(theta):
    # norm.pdf 正态分布的概率密度函数
    # loc是均值，scale 是标准差
    y = norm.pdf(theta, loc=3, scale=2)
    return y


T = 10000
pi = [0 for i in range(T)]
sigma = 1
# 设置初始值
t = 0
# 遍历执行
while t < T - 1:
    t = t + 1
    # 状态转移进行随机抽样
    # norm.rvs 生成服从指定分布的随机数
    # norm.rvs通过loc和scale参数可以指定随机变量的偏移和缩放参数，
    # 这里对应的是正态分布的期望和标准差。size得到随机数数组的形状参数。
    # (也可以使用np.random.normal(loc=0.0, scale=1.0, size=None))
    pi_star = norm.rvs(loc=pi[t - 1], scale=sigma, size=1, random_state=None)
    # 计算接受概率
    alpha = min(1, (smooth_dist(pi_star[0]) / smooth_dist(pi[t - 1])))
    # 从均匀分布中随机抽取一个数u
    u = random.uniform(0, 1)
    # 拒绝-接受采样
    if u < alpha:
        pi[t] = pi_star[0]
    else:
        pi[t] = pi[t - 1]

# 绘制采样分布
plt.scatter(pi, norm.pdf(pi, loc=3, scale=2), label='Target Distribution')
num_bins = 100
plt.hist(pi,
         num_bins,
         density=1,
         facecolor='red',
         alpha=0.6,
         label='Samples Distribution')
plt.legend()
plt.show()

# Gibbs


import math
# 导入多元正态分布函数
from scipy.stats import multivariate_normal

# 指定二元正态分布均值和协方差矩阵
samplesource = multivariate_normal(mean=[5, -1], cov=[[1, 0.5], [0.5, 2]])


# 定义给定x的条件下y的条件状态转移分布
def p_yx(x, m1, m2, s1, s2):
    return (random.normalvariate(m2 + rho * s2 / s1 * (x - m1), math.sqrt(1 - rho ** 2) * s2))


# 定义给定y的条件下x的条件状态转移分布
def p_xy(y, m1, m2, s1, s2):
    return (random.normalvariate(m1 + rho * s1 / s2 * (y - m2), math.sqrt(1 - rho ** 2) * s1))


# 指定相关参数
N, K = 5000, 20
x_res = []
y_res = []
z_res = []
m1, m2 = 5, -1
s1, s2 = 1, 2
rho, y = 0.5, m2

# 遍历迭代
for i in range(N):
    for j in range(K):
        # y给定得到x的采样
        x = p_xy(y, m1, m2, s1, s2)
        # x给定得到y的采样
        y = p_yx(x, m1, m2, s1, s2)
        # 对于上面创建的对象 samplesource 也可以调用 pdf 方法，随机抽样
        z = samplesource.pdf([x, y])
        x_res.append(x)
        y_res.append(y)
        z_res.append(z)
# 绘图
num_bins = 50
plt.hist(x_res, num_bins, density=1, facecolor='green', alpha=0.5, label='x')
plt.hist(y_res, num_bins, density=1, facecolor='red', alpha=0.5, label='y')
plt.title('Histogram')
plt.legend()
plt.show()

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=30, azim=20)
ax.scatter(x_res, y_res, z_res, marker='o')
plt.show()
