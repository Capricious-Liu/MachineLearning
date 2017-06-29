import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf

# 在运行计算时输入该值  x不是特定的值，只是一个占位符
# None 表示第一维可以是任何长度
x = tf.placeholder(tf.float32, [None, 784])

# 784行 10列
# 10行
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W)+b)

# y_ 真实值
# tf.log() 计算每一个元素的对数
# * ： —y_ 和 tf.log(y) 中对应元素相乘
# tf.reduce_sum  计算张量的所有元素和
y_ = tf.placeholder("float", [None, 10])
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_ : batch_ys})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuary = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print(sess.run(accuary,feed_dict = {x: mnist.test.images, y_:mnist.test.labels}))