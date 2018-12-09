import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os

for i in os.listdir('log/'):
    os.remove('log/' + i)

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
test_x = mnist.test.images[:2000]
test_y = mnist.test.labels[:2000]


def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result


def weight_variable(shape):
    inital = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    tf.summary.histogram('w', inital)
    return inital


def bias_variable(shape):
    inital = tf.Variable(tf.constant(0.1, shape=shape))
    tf.summary.histogram('b', inital)
    return inital


def conv2d(x, W):
    # stride[1,x_movement,ymovement,1]
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def conv2(inputs, filters, input_ch, padding, kernel_size):
    # channels_in = inputs[0, 0, 0, :].get_shape().as_list()[0]
    with tf.name_scope('conv2d'):
        W = tf.Variable(tf.truncated_normal(shape=[kernel_size, kernel_size, input_ch, filters], stddev=0.1,  name='W'))
        # W = tf.Variable(tf.constant(1.0, shape=[kernel_size, kernel_size, channels_in, filters]), name='W')
        # W = tf.get_variable(initializer=tf.constant(0.1, shape=[kernel_size, kernel_size, channels_in, filters]))
        b = tf.Variable(tf.constant(0.1, shape=[filters]), name='b')
        conv = tf.nn.conv2d(inputs, W, strides=[1, 1, 1, 1], padding=padding)
        sum = conv + b
        act = tf.nn.relu(conv + b)
    tf.summary.histogram('conv', conv)
    tf.summary.histogram('add_b', sum)
    tf.summary.histogram('weights', W)
    tf.summary.histogram('biases', b)
    tf.summary.histogram('activations', act)
    return act

def max_pool_2(x):
    # stride[1,x_movement,ymovement,1]
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')


with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 784])
    ys = tf.placeholder(tf.float32, [None, 10])
    keep_prob = tf.placeholder(tf.float32)

x_img = tf.reshape(xs, [-1, 28, 28, 1])
# print(x_img.shape)
tf.summary.image('input', x_img)
# conv1
with tf.name_scope('layer'):
    w_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_img, w_conv1) + b_conv1)
    #h_conv1 = conv2(inputs=x_img, filters=32, input_ch=1, padding='SAME', kernel_size=5)
    h_pool1 = max_pool_2(h_conv1)

# conv
with tf.name_scope('layer'):
    w_conv2 = weight_variable([3, 3, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
    #h_conv2 = conv2(inputs=h_pool1, filters=64, input_ch=32, padding='SAME', kernel_size=3)
    h_pool2 = max_pool_2(h_conv2)

# func1
with tf.name_scope('layer'):
    w_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# func2
with tf.name_scope('layer'):
    w_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    output = tf.matmul(h_fc1_drop, w_fc2) + b_fc2
    prediction = tf.nn.softmax(output)
# error
with tf.name_scope('loss'):
    loss = tf.losses.softmax_cross_entropy(onehot_labels=ys, logits=output)
tf.summary.scalar('loss', loss)
with tf.name_scope('train'):
    train = tf.train.AdamOptimizer(1e-3).minimize(loss)

merged = tf.summary.merge_all()
# sess = tf.Session()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('log/', sess.graph)
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        # print(batch_xs.shape)
        _, totle_loss, summary = sess.run([train, loss, merged],
                                          feed_dict={
                                          xs: batch_xs, ys: batch_ys, keep_prob: 1})
        # print(totle_loss)
        if i % 1 == 0:
            print('Step: ' + str(i) + ' | accuracy: ' +
                  str(compute_accuracy(test_x, test_y)) +
                  ' | loss: ' + str(totle_loss))
            writer.add_summary(summary, i)
            # print('loss: ' + str(totle_loss))
