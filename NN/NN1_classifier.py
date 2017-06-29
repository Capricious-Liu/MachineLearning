import tensorflow as tf
import numpy as np
from sklearn import preprocessing
# 定义参数
n_hidden_1 = 5  # 第一层神经元
n_hidden_2 = 2  # 第二层神经元
n_input = 25  # 输入大小
n_classes = 1  # 结果是要得到一个几分类的任务


# 输入和输出
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

# 权重和偏置参数
stddev = 0.1
# weights = {
#     'w1': tf.Variable(tf.random_normal([n_input, n_hidden_1], stddev=stddev)),
#     'w2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], stddev=stddev)),
#     'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes], stddev=stddev))
# }
# biases = {
#     'b1': tf.Variable(tf.random_normal([n_hidden_1])),
#     'b2': tf.Variable(tf.random_normal([n_hidden_2])),
#     'out': tf.Variable(tf.random_normal([n_classes]))
# }
weights = {
    'w1': tf.Variable(tf.truncated_normal([n_input, n_hidden_1], mean=0.5, stddev=stddev)),
    'w2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2], mean=0.5, stddev=stddev)),
    'out': tf.Variable(tf.truncated_normal([n_hidden_2, n_classes], mean=0.5, stddev=stddev))
}
biases = {
    'b1': tf.Variable(tf.truncated_normal([n_hidden_1], mean=0.5, stddev=stddev)),
    'b2': tf.Variable(tf.truncated_normal([n_hidden_2], mean=0.5, stddev=stddev)),
    'out': tf.Variable(tf.truncated_normal([n_classes], mean=0.5, stddev=stddev))
}
print("NETWORK READY")


def multilayer_perceptron(_X, _weights, _biases):
    # 第1层神经网络 = tf.nn.激活函数(tf.加上偏置量(tf.矩阵相乘(输入Data, 权重W1), 偏置参数b1))
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(_X, _weights['w1']), _biases['b1']))
    # 第2层的格式与第1层一样，第2层的输入是第1层的输出。
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, _weights['w2']), _biases['b2']))
    # 返回预测值
    y_ = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, _weights['out']), _biases['out']))

    # return (tf.matmul(layer_2, _weights['out']) + _biases['out'])
    # y_ = 1.0/(1.0 + tf.exp(-(tf.matmul(X, w))))
    return y_


# 预测
pred = multilayer_perceptron(x, weights, biases)
pred_output = tf.round(pred)
# 计算损失函数和优化
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y, name='xentropy'),name='xentropy')
# cost = -tf.reduce_sum(Y*tf.log(py_x) + (1 - Y) * tf.log(1 - py_x))
# cost = -tf.reduce_mean(y * tf.log(pred))
# cost = -tf.reduce_sum(y*tf.log(tf.clip_by_value(pred, 1e-10, 1.0)))
cost = -tf.reduce_sum(y*tf.log(pred) + (1 - y) * tf.log(1 - pred))
optm = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)


accr = np.sum(np.sum(np.abs(pred_output - y))) / 900
# corr = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
# accr = tf.reduce_mean(tf.cast(corr, "float"))


# 初始化
init = tf.global_variables_initializer()
print("FUNCTIONS READY")

# 训练
training_epochs = 80
batch_size = 40
display_step = 4
# LAUNCH THE GRAPH
sess = tf.Session()
sess.run(init)

import pandas as pd
import numpy as np
# data = pd.read_csv("../TrainData_temp.csv",usecols=[0,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43],encoding='gb2312')
# data.fillna(value=-1, inplace=True)
# label = pd.read_csv("../TrainData_temp.csv", usecols=[46], encoding='gb2312')
# label = np.array(label)
# X = np.array(data)
# X = X[:, 1:40]
# # X.append([3])
# label2 = np.ones(label.shape)
# label2 = label2 - label
# label = np.c_[label, label2]
# # lb = preprocessing.LabelBinarizer()
# # label = lb.fit_transform(label)



# read data needed implemetation
data = pd.read_csv("splitedData.csv", usecols=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25])
data.fillna(value=-1, inplace=True)
label = pd.read_csv("Shuffled.csv", usecols=[40])
label = np.array(label)
# label2 = np.ones(label.shape)
# label2 = label2 - label
# label = np.c_[label, label2]

X = np.array(data)


max_items = X.shape[0]
saver = tf.train.Saver()
# 优化器
for epoch in range(training_epochs):
    avg_cost = 0.
    total_batch = int(max_items / batch_size)+1  # 计算total batch
    # 迭代训练
    for i in range(total_batch):
        # batch_xs, batch_ys = mnist.train.next_batch(batch_size)  # 输入
        batch_xs = X[batch_size*i:batch_size*(i+1)]
        batch_ys = label[batch_size*i:batch_size*(i+1)]

        feeds = {x: batch_xs, y: batch_ys}
        sess.run(optm, feed_dict=feeds)

        avg_cost += sess.run(cost, feed_dict=feeds)
    avg_cost = avg_cost / total_batch
    # 打印结果
    if (epoch + 1) % display_step == 0:
        print("Epoch: %03d/%03d cost: %.9f" % (epoch, training_epochs, avg_cost))
        pred_temp = sess.run(pred_output, feed_dict={x:X, y:label})
        # print(pred_temp)
        # feeds = {x: batch_xs, y: batch_ys}
        # train_acc = sess.run(accr, feed_dict=feeds)
        # print("TRAIN ACCURACY: %.3f" % (train_acc))
        # feeds = {x: X, y: label}
        # test_acc = sess.run(accr, feed_dict=feeds)

        test_acc = np.sum(np.sum(np.abs(pred_temp - y))) / 900
        print("TEST ACCURACY: ", test_acc)

        w_lay1, b_lay1 = sess.run([weights['out'], biases['out']])
        # print(w_lay1)
        # print("!!!!!!!")
        # print(b_lay1)
        # print("##########################")

        saver.save(sess, "../result/TrainData.csv", global_step=epoch)
print("OPTIMIZATION FINISHED")