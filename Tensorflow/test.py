import tensorflow as tf
# matrix1 = tf.constant([[3, 3.]])
# matrix2 = tf.constant([[2.], [2.]])
#
# product = tf.matmul(matrix1,matrix2)
#
# sess = tf.Session()
#
# result = sess.run(product)
# print(result)
#
# sess.close()

# state = tf.Variable(0, name='counter')
#
# one = tf.constant(1)
# new_value = tf.add(state, one)
# update = tf.assign(state, new_value)
#
# # variables should be initialized before
# init_op = tf.global_variables_initializer()
#
# with tf.Session() as sess:
#     sess.run(init_op)
#     print(sess.run(state))
#
#     for _ in range(3):
#         result = sess.run(update)
#         print(result)

# Fetch
input1 = tf.constant(3.0)
input2 = tf.constant(2.0)
input3 = tf.constant(5.0)
intermed = tf.add(input2, input3)
mul = tf.multiply(input1, intermed)

with tf.Session() as sess:
    result = sess.run([mul, intermed])
    print(result)