import tensorflow as tf

if __name__ == '__main__':
    a = tf.placeholder(tf.float32)
    b = tf.placeholder(tf.float32)
    adder_node = tf.add(a, b)
    mult_node = tf.multiply(adder_node, 10)

    sess = tf.Session()
    tf.summary.FileWriter('logs', sess.graph)
    print(sess.run(mult_node, {a: 3, b: 4.5}))
    print(sess.run(mult_node, {a: [3, 1.7], b: [4.5, 19.7]}))
