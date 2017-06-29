import tensorflow as tf
import numpy as np
import pandas as pd
import sys

from sklearn.metrics import roc_auc_score


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def init_biases(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def model(X,w):
    return tf.matmul(X,w)

batch_size = 20
origin_learning_rate = 0.001
num_iters = 30000

trX = pd.read_csv("splitedData.csv", usecols=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25])

trX.fillna(value=-1, inplace=True)
trY = pd.read_csv("Shuffled.csv", usecols=[40])
trY = np.array(trY)
trX = np.array(trX)

teX = trX
teY = trY


print("finish loading train set ")


num_features = trX[0].shape[0]

print('num_features: ', num_features)
print('trainSet size: ', len(trX), '      ',len(trY))
print('testSet size: ', len(teX))


X = tf.placeholder("float", [None, num_features])  # create symbolic variables

Y = tf.placeholder("float", [None, 1])

current_iter = tf.Variable(0)

w = init_weights([num_features, 1])  # like in linear regression, we need a shared variable weight matrix for logistic regression
# b = init_biases([1])
# py_x = model(X, w, b)
py_x = model(X, w)

learning_rate = tf.train.exponential_decay(origin_learning_rate, current_iter, decay_steps= num_iters, decay_rate = 0.3)

cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=py_x, labels=Y))

train_op = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost, global_step=current_iter)  # construct optimizer

predict_op =tf.round(tf.nn.sigmoid(py_x))


sess = tf.Session()

init = tf.initialize_all_variables()

sess.run(init)

for i in range(num_iters):
    current_iter = i
    predicts, cost_ = sess.run([predict_op, cost], feed_dict={X: teX, Y: teY})
    # print(i, 'auc:', roc_auc_score(teY, predicts), 'cost:', cost_)
    print(i, 'auc:', np.sum(np.abs(teY -predicts))/900, 'cost:', cost_)
    # print(predicts)
    # print(teY)
    for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX), batch_size)):
        sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})
        # predicts, cost_ = sess.run([predict_op, cost], feed_dict={X: trX[start:end], Y: trY[start:end]})
        # print(start, 'auc:', roc_auc_score(trY[start:end], predicts), 'cost:', cost_)

print('final ', 'auc:', roc_auc_score(teY, predicts), 'cost:', cost_)
