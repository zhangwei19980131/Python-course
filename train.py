import pandas as pd
import tensorflow as tf
import numpy as np

# ——————————————————导入数据——————————————————————
f = open('股票数据.csv')  # 重新写入数据位置和名称等
df = pd.read_csv(f)  # 读入股票数据
data = df.iloc[:, 2:10].values  # 取第3-10列


# 获取训练集，从0到5800个，即是前多少行数据
def get_train_data(batch_size=60, time_step=20, train_begin=0, train_end=5800):  # 函数传值，
    batch_index = []
    data_train = data[train_begin:train_end]  # 训练数据开始至结束
    normalized_train_data = (data_train - np.mean(data_train, axis=0)) / np.std(data_train, axis=0)  # 定义标准化语句
    train_x, train_y = [], []  # 训练集
    for i in range(len(normalized_train_data) - time_step):  # 以下即是获取训练集并进行标准化，并返回该函数的返回值
        if i % batch_size == 0:
            batch_index.append(i)
        x = normalized_train_data[i:i + time_step, :7]  # 即为前7列为输入维度数据
        y = normalized_train_data[i:i + time_step, 7, np.newaxis]  # 最后一列标签为Y，可以说出是要预测的，并与之比较，反向求参
        train_x.append(x.tolist())
        train_y.append(y.tolist())
    batch_index.append((len(normalized_train_data) - time_step))
    return batch_index, train_x, train_y


# ——————————————————定义神经网络变量——————————————————
# 输入层、输出层权重、偏置

rnn_unit=10       #hidden layer units

input_size=1      #输入层维度
output_size=1     #输出层维度
weights = {
    'in': tf.Variable(tf.random.normal([input_size, rnn_unit])),
    'out': tf.Variable(tf.random.normal([rnn_unit, 1]))
}
biases = {
    'in': tf.Variable(tf.constant(0.1, shape=[rnn_unit, ])),
    'out': tf.Variable(tf.constant(0.1, shape=[1, ]))
}


# ——————————————————定义神经网络变量——————————————————
def lstm(X):
    batch_size = tf.shape(X)[0]
    time_step = tf.shape(X)[1]
    w_in = weights['in']
    b_in = biases['in']
    input = tf.reshape(X, [-1, input_size])  # 需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
    input_rnn = tf.matmul(input, w_in) + b_in
    input_rnn = tf.reshape(input_rnn, [-1, time_step, rnn_unit])  # 将tensor转成3维，作为lstm cell的输入
    cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
    init_state = cell.zero_state(batch_size, dtype=tf.float32)
    output_rnn, final_states = tf.nn.dynamic_rnn(cell, input_rnn, initial_state=init_state, dtype=tf.float32)
    output = tf.reshape(output_rnn, [-1, rnn_unit])
    w_out = weights['out']
    b_out = biases['out']
    pred = tf.matmul(output, w_out) + b_out
    return pred, final_states


# ————————————————训练模型————————————————————

def train_lstm(batch_size=60, time_step=20, train_begin=2000, train_end=5800):
    X = tf.placeholder(tf.float32, shape=[None, time_step, input_size])  # 预先定义X,Y占位符
    Y = tf.placeholder(tf.float32, shape=[None, time_step, output_size])
    batch_index, train_x, train_y = get_train_data(batch_size, time_step, train_begin, train_end)
    pred, _ = lstm(X)
    loss = tf.reduce_mean(tf.square(tf.reshape(pred, [-1]) - tf.reshape(Y, [-1])))  # 定义损失函数
    train_op = tf.train.AdamOptimizer(lr).minimize(loss)  # 定义优化
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=15)  # 保存模型

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())  # 做一次全局sess
        for i in range(10):  # 这个迭代次数，可以更改，越大预测效果会更好，但需要更长时间
            for step in range(len(batch_index) - 1):  # 喂数据
                _, loss_ = sess.run([train_op, loss], feed_dict={X: train_x[batch_index[step]:batch_index[step + 1]],
                                                                 Y: train_y[batch_index[step]:batch_index[step + 1]]})
            print("Number of iterations:", i, " loss:", loss_)
        print("The train has finished")


train_lstm()
