import numpy as np
import cv2
import matplotlib.pyplot as plt

# n是迭代次数
def kmeans(n, k, image):
    height = image.shape[0]
    width = image.shape[1]
    tmp = image.reshape(-1, 3)
    result = tmp.copy()

    # 扩展一个维度用来存放标签
    result = np.column_stack((result, np.ones(height * width)))

    center_point = np.random.choice(height * width, k, replace=False)
    center = result[center_point, :]
    distance = [[] for i in range(k)]

    # 迭代
    for i in range(n):
        for j in range(k): # 计算欧式距离
            distance[j] = np.sqrt(np.sum(np.square(result - np.array(center[j])), axis=1))
        result[:, 3] = np.argmin(np.array(distance), axis=0)
        for j in range(k):
            center[j] = np.mean(result[result[:, 3] == j], axis=0) # 求均值
    return result


if __name__ == '__main__':
    img = cv2.imread('lena.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.subplot(121)
    plt.imshow(img)
    height = img.shape[0]
    width = img.shape[1]

    result_img = kmeans(150, 4, img)
    result_img = result_img[:, 3].reshape(height, width)

    plt.subplot(122)
    plt.imshow(result_img,'gray')
    plt.show()
