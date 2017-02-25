import tensorflow as tf

if __name__ == '__main__':
    # build a graph
    node1 = tf.constant(3.0, tf.float32)
    node2 = tf.constant(4.0)

    # graph is not evaluated yet
    print(node1, node2)

    # run the graph to evaluate the result
    sess = tf.Session()
    result = sess.run([node1, node2])
    print(result)
