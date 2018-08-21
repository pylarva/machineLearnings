# -*- coding: utf-8 -*-


import tensorflow as tf


labels = [[0,0,1],[0,1,0]]
logits = [[2,  0.5,6],
          [0.1,0,  3]]
logits_scaled = tf.nn.softmax(logits)
logits_scaled2 = tf.nn.softmax(logits_scaled)

# 正确的softmax_cross_entropy_with_logits使用方式
result1 = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
# 传入softmax_cross_entropy_with_logits的logits不能是已经进行softmax过的
result2 = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits_scaled)
# 自己写的组合公式计算交叉熵
result3 = -tf.reduce_sum(labels*tf.log(logits_scaled), 1)


with tf.Session() as sess:
    print("scaled=",sess.run(logits_scaled))
    # 经过第二次的softmax后，分布概率会有变化
    print("scaled2=",sess.run(logits_scaled2))

    # 正确的方式
    print("rel1=",sess.run(result1),"\n")
    # 如果将softmax变换完的值放进去会，就相当于算第二次softmax的loss，所以会出错
    print("rel2=",sess.run(result2),"\n")
    print("rel3=",sess.run(result3))


# 标签总概率为1
labels = [[0.4, 0.1, 0.5], [0.3, 0.6, 0.1]]
result4 = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
with tf.Session() as sess:
    print("rel4=", sess.run(result4), "\n")

# sparse
# 其实是0 1 2 三个类。等价 第一行 001 第二行 010
labels = [2, 1]
result5 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
with tf.Session() as sess:
    print("rel5=", sess.run(result5),"\n")
    
# 注意！！！这个函数的返回值并不是一个数，而是一个向量，
# 如果要求交叉熵loss，我们要对向量求均值，
# 就是对向量再做一步tf.reduce_sum操作
loss = tf.reduce_sum(result1)
with tf.Session() as sess:
    print("loss=", sess.run(loss))

# 对于已经求得softmax的情况求loss 可以简化成如下
labels = [[0, 0, 1], [0, 1, 0]]
loss2 = -tf.reduce_sum(labels * tf.log(logits_scaled))    
with tf.Session() as sess:
    print("loss2=", sess.run(loss2))