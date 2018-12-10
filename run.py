import argparse
import tensorflow as tf
import sys
import time
import logging
import cv2
import numpy as np
from tensorflow.contrib import slim
import vgg
from cpm import CpmStage1
import common
from tensblur.smoother import Smoother
from estimator import PoseEstimator, TfPoseEstimator

logger = logging.getLogger('run')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training codes for Openpose using Tensorflow')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints/train/2018-12-9-22-31-25')
    parser.add_argument('--backbone_net_ckpt_path', type=str, default='checkpoints/vgg/vgg_19.ckpt')
    parser.add_argument('--img_path', type=str, default='images/1.png')
    parser.add_argument('--run_model', type=str, default='webcam')
    args = parser.parse_args()

    checkpoint_path = args.checkpoint_path
    logger.info('checkpoint_path: ' + checkpoint_path)

    with tf.name_scope('inputs'):
        raw_img = tf.placeholder(tf.float32, shape=[None, None, None, 3])
        img_size = tf.placeholder(dtype=tf.int32, shape=(2,), name='original_image_size')

    img_normalized = raw_img / 255 - 0.5

    # define vgg19
    with slim.arg_scope(vgg.vgg_arg_scope()):
        vgg_outputs, end_points = vgg.vgg_19(img_normalized)

    # get net graph
    logger.info('initializing model...')
    net = CpmStage1(inputs_x=vgg_outputs)
    hm_pre, cpm_pre, added_layers_out = net.gen_net()

    hm_up = tf.image.resize_area(hm_pre[4], img_size)
    cpm_up = tf.image.resize_area(cpm_pre[4], img_size)
    smoother = Smoother({'data': hm_up}, 25, 3.0)
    gaussian_heatMat = smoother.get_output()

    max_pooled_in_tensor = tf.nn.pool(gaussian_heatMat, window_shape=(3, 3), pooling_type='MAX', padding='SAME')
    tensor_peaks = tf.where(tf.equal(gaussian_heatMat, max_pooled_in_tensor), gaussian_heatMat,
                                 tf.zeros_like(gaussian_heatMat))

    logger.info('initialize saver...')
    trainable_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='train_layers')
    restorer = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='vgg_19'), name='vgg_restorer')
    saver = tf.train.Saver(trainable_var_list)

    logger.info('initialize session...')
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.group(tf.global_variables_initializer()))
        logger.info('restoring vgg weights...')
        restorer.restore(sess, args.backbone_net_ckpt_path)
        logger.info('restoring from checkpoint...')
        saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir=checkpoint_path))
        logger.info('initialization done')

        if args.run_model == 'webcam':
            cap = cv2.VideoCapture('http://admin:admin@192.168.1.103:8081')
            # cap = cv2.VideoCapture(0)
            _, image = cap.read()
            if image is None:
                logger.error('Image can not be read')
                sys.exit(-1)
            size = [image.shape[1], image.shape[0]]
            h = int(480 * (image.shape[0] / image.shape[1]))
            while True:
                _, image = cap.read()
                img = np.array(cv2.resize(image, (h, 480)))
                img = img[np.newaxis, :]
                peaks, heatmap, vectormap = sess.run([tensor_peaks, hm_up, cpm_up],
                                                     feed_dict={raw_img: img, img_size: size})
                bodys = PoseEstimator.estimate_paf(peaks[0], heatmap[0], vectormap[0])
                image = TfPoseEstimator.draw_humans(image, bodys, imgcopy=False)
                cv2.imshow(' ', image)
                cv2.waitKey(1)
        else:
            image = common.read_imgfile(args.img_path)
            size = [image.shape[0], image.shape[1]]
            if image is None:
                logger.error('Image can not be read, path=%s' % args.img_path)
                sys.exit(-1)
            img = np.array(cv2.resize(image, (320, 240)))
            img = img[np.newaxis, :]
            peaks, heatmap, vectormap = sess.run([tensor_peaks, hm_up, cpm_up], feed_dict={raw_img: img, img_size: size})
            bodys = PoseEstimator.estimate_paf(peaks[0], heatmap[0], vectormap[0])
            image = TfPoseEstimator.draw_humans(image, bodys, imgcopy=False)
            cv2.imshow(' ', image)
            cv2.waitKey(0)
