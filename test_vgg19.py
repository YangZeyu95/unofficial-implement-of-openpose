import tensorflow as tf
from tensorflow.contrib import slim
import vgg
import cv2
import numpy as np

img = cv2.imread("images/1.png")
img = cv2.resize(img, (368, 368))
img = np.resize(img, [1, 368, 368, 3])
vgg19_ckpt_path = "checkpoints/vgg/vgg_19.ckpt"

with tf.name_scope('inputs'):
    inputs = tf.placeholder(tf.float32, shape=(None, None, None, 3))
with slim.arg_scope(vgg.vgg_arg_scope()):
    vgg_outputs, end_points = vgg.vgg_19(inputs)

restorer = tf.train.Saver()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True





with tf.Session(config=config) as sess:
    restorer.restore(sess, vgg19_ckpt_path)
    print("model restored")
    predict = sess.run(vgg_outputs, feed_dict={inputs: img})
    for i in range(512):
        cv2.imshow(" ", predict[0, :, :, i-1])
        cv2.waitKey(0)
    # print(vgg_outputs.shape)

