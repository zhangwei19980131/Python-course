import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# 训练和预测
only_prediction = 0

# 数据常量
colnums = 21
output_size = 10
rows = 3000
# 神经网络常量
rnn_unit = 20  # hidden layer units
input_size = colnums - output_size
batch_size = 5
time_step = 10
epoch = 2000
lr = 0.0006  # 学习率
# 其他常数
# model_path='./model'
model_path = './model2'
data_path = './dataset.csv'

# 定义输入输出的w和b
wi = tf.Variable(tf.random.normal([input_size, rnn_unit]))
bi = tf.Variable(tf.random.normal(shape=[rnn_unit, ], mean=0.0, stddev=1.0))
wo = tf.Variable(tf.random.normal([rnn_unit, output_size]))
bo = tf.Variable(tf.random.normal(shape=[output_size, ], mean=0.0, stddev=1.0))


# 导入数据
def import_data():
    f = open('股票数据.csv')
    df = pd.read_csv(f)  # 读入股票数据
    data = df.iloc[:rows, :colnums].values
    return data


# 获取训练集
def get_train_data(data, train_begin=0, train_end=500):
    batch_index = []
    data_train = data[train_begin:train_end]
    train_x, train_y = [], []  # 训练集
    for i in range(int(len(data_train) / time_step)):
        if i % batch_size == 0:
            batch_index.append(i)
        data = data_train[i * time_step:(i + 1) * time_step, :]  # 对每批中每组数据进行标准化
        x = data[:, :colnums - output_size]
        mean_x = np.mean(x, axis=0)
        std_x = np.std(x, axis=0) + 0.1
        x = (x - mean_x) / std_x
        y = data[:, colnums - output_size:colnums]
        mean_y = np.mean(y, axis=0)
        std_y = np.std(y, axis=0) + 0.1
        y = (y - mean_y) / std_y
        train_x.append(x.tolist())
        train_y.append(y.tolist())
    return batch_index, train_x, train_y


# 获取测试集
def get_test_data(data, test_begin=0, test_end=20):
    data_test = data[test_begin:test_end]

    test_x, test_y, mean, std = [], [], [], []
    for i in range(int(len(data_test) / time_step)):
        x = data_test[i * time_step:(i + 1) * time_step, :colnums - output_size]
        mean_x = np.mean(x, axis=0)
        std_x = np.std(x, axis=0) + 0.1
        x = (x - mean_x) / std_x
        mean_y, std_y = [], []
        for j in (range(output_size)):
            mean_y.append(mean_x[0])
            std_y.append(std_x[0])
        mean.append(mean_y)
        std.append(std_y)
        test_x.append(x.tolist())
    test_y = data_test[:, 0]
    return test_x, test_y, mean, std


# 定义模型
def RNN(cell, X, init_state):
    # 定义隐藏层的运算
    input = tf.reshape(X, [-1, input_size])
    input_rnn = tf.matmul(input, wi) + bi

    input_rnn = tf.reshape(input_rnn, [-1, time_step, rnn_unit])
    output_rnn, final_states = tf.nn.dynamic_rnn(cell, input_rnn, initial_state=init_state,
                                                 dtype=tf.float32)  # output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果
    output = tf.reshape(output_rnn, [-1, rnn_unit])  # 作为输出层的输入

    pred = tf.matmul(output, wo) + bo
    return pred, final_states


# 开始训练
def train_data(train_begin=500, train_end=2500):
    # 读数据集,设置神经网络模型,准备X和Y,初始状态
    batch_index, train_x, train_y = get_train_data(import_data(), train_begin, train_end)
    cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
    X = tf.placeholder(tf.float32, shape=[None, time_step, input_size])
    Y = tf.placeholder(tf.float32, shape=[None, time_step, output_size])
    init_state = cell.zero_state(batch_size, dtype=tf.float32)

    # 构建计算图结点
    pred, final_states = RNN(cell, X, init_state)
    loss = tf.reduce_mean(tf.square(tf.reshape(pred, [-1, ]) - tf.reshape(Y, [-1, ])))
    train_op = tf.train.AdamOptimizer(lr).minimize(loss)

    # 声明保存模型需要用的对象
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=15)

    # 开始训练
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for i in range(epoch):
        for step in range(len(batch_index) - 1):
            feed_dict = {X: train_x[batch_index[step]:batch_index[step + 1]],
                         Y: train_y[batch_index[step]:batch_index[step + 1]]}

            _, loss_, states = sess.run([train_op, loss, final_states],
                                        feed_dict=feed_dict)
        print(i, loss_)
        if i % 20 == 0:
            print("保存模型：", saver.save(sess, model_path + '/stock.model', global_step=i))
    print("保存模型：", saver.save(sess, model_path, global_step=i))
    sess.close()


# 预测
def prediction(test_begin=0, test_end=500):
    # 读数据集,配置X和初始状态
    test_x, test_y, mean, std = get_test_data(import_data(), test_begin, test_end)
    X = tf.placeholder(tf.float32, shape=[None, time_step, input_size])
    cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)

    init_state = cell.zero_state(1, dtype=tf.float32)

    # 构建计算图结点
    pred, final_states = RNN(cell, X, init_state)

    # 配置读取模型需要用的对象
    saver = tf.train.Saver(tf.global_variables())

    # 开始预测
    sess = tf.Session()
    module_file = tf.train.latest_checkpoint(model_path)
    saver.restore(sess, module_file)
    test_predict = []

    for step in range(len(test_x)):
        prob = sess.run(pred, feed_dict={X: [test_x[step]]})
        test_predict.append(prob[len(prob) - 1])
    sess.close
    # acc = np.average(np.abs(test_predict - test_y) / test_y)  # 偏差
    # print(acc)
    # 以折线图表示结果
    real = []
    pred = (np.array(test_predict) * std) + mean
    plt.figure()
    plt.plot(list(range(len(test_y))), test_y, color='r')
    for step in range(0, len(test_predict) - 1):
        real.append(test_y[(step + 1) * time_step:(step + 1) * time_step + output_size])
        plt.plot(list(range((step + 1) * time_step, (step + 1) * time_step + output_size)), pred[step], color='b',
                 marker="v")
    acc = np.average(np.abs(pred[:len(real)] - np.array(real)) / np.array(real))  # 偏差
    print("acc", acc)
    plt.show()


if only_prediction == 0:
    with tf.variable_scope('train'):
        train_data(500, 2000)
    with tf.variable_scope('train', reuse=True):
        prediction(2500, 3000)
else:
    with tf.variable_scope('train'):
        prediction(2500, 3000)
