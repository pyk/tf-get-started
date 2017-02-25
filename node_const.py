import tensorflow as tf

if __name__ == '__main__':
    # build a graph
    node1 = tf.constant(3.0, tf.float32)
    node2 = tf.constant(4.0)

    # graph is not evaluated yet
    print('node1:', node1)
    print('node2:', node2)

    # Add ops to the graph
    sumnode = tf.add(node1, node2)
    print('sumnode:', sumnode)

    # run the graph to evaluate the result
    sess = tf.Session()
    result = sess.run([sumnode])
    print(result)

    # Summary Writer
    summary_writer = tf.summary.FileWriter('logs', sess.graph)
