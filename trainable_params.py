import tensorflow as tf

if __name__ == '__main__':
    # Model parameters
    W = tf.Variable([.3], tf.float32)
    b = tf.Variable([1.0], tf.float32)

    # Model input & output
    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)
    linear_model = (W * x) + b

    # Model loss function
    squared_deltas = tf.square(linear_model - y)
    loss = tf.reduce_sum(squared_deltas)

    # Model optimizer
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss)

    # Training data
    x_data = [0.5, 0.7, 0.5, 0.66, 0.8, 0.45, 0.3, 0.2, 0.0, 0.12]
    y_data = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

    # training loop
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    for i in range(10000):
        sess.run(train, {x: x_data, y: y_data})
    tf.summary.FileWriter('logs', sess.graph)

    # evaluate training accuracy
    curr_W, curr_b, curr_loss  = sess.run([W, b, loss], 
        {x:x_data, y:y_data})
    print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
