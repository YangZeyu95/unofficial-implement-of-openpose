import argparse
import tensorflow as tf
import sys
import time
import logging
import cv2
import numpy as np
from tensorflow.contrib import slim
import vgg
from cpm import PafNet
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
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints/train/2018-12-13-16-56-49/')
    parser.add_argument('--backbone_net_ckpt_path', type=str, default='checkpoints/vgg/vgg_19.ckpt')
    parser.add_argument('--image', type=str, default=None)
    # parser.add_argument('--run_model', type=str, default='img')
    parser.add_argument('--video', type=str, default=None)
    parser.add_argument('--train_vgg', type=bool, default=True)
    parser.add_argument('--use_bn', type=bool, default=False)
    parser.add_argument('--save_video', type=str, default='result/our.mp4')

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
    net = PafNet(inputs_x=vgg_outputs, use_bn=args.use_bn)
    hm_pre, cpm_pre, added_layers_out = net.gen_net()

    hm_up = tf.image.resize_area(hm_pre[5], img_size)
    cpm_up = tf.image.resize_area(cpm_pre[5], img_size)
    # hm_up = hm_pre[5]
    # cpm_up = cpm_pre[5]
    smoother = Smoother({'data': hm_up}, 25, 3.0)
    gaussian_heatMat = smoother.get_output()

    max_pooled_in_tensor = tf.nn.pool(gaussian_heatMat, window_shape=(3, 3), pooling_type='MAX', padding='SAME')
    tensor_peaks = tf.where(tf.equal(gaussian_heatMat, max_pooled_in_tensor), gaussian_heatMat,
                                 tf.zeros_like(gaussian_heatMat))

    logger.info('initialize saver...')
    # trainable_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='openpose_layers')
    # trainable_var_list = []
    trainable_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='openpose_layers')
    if args.train_vgg:
        trainable_var_list = trainable_var_list + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='vgg_19')

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
        # saver.restore(sess, args.checkpoint_path + 'model-55000.ckpt')
        logger.info('initialization done')
        if args.image is None:
            if args.video is not None:
                cap = cv2.VideoCapture(args.video)
            else:
                cap = cv2.VideoCapture(0)
                cap = cv2.VideoCapture('http://admin:admin@192.168.1.52:8081')
            _, image = cap.read()
            if image is None:
                logger.error("Can't read video")
                sys.exit(-1)
            fps = cap.get(cv2.CAP_PROP_FPS)
            ori_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            ori_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if args.save_video is not None:
                fourcc = cv2.VideoWriter_fourcc(*'MP4V')
                video_saver = cv2.VideoWriter('result/our.mp4', fourcc, fps, (ori_w, ori_h))
                logger.info('record vide to %s' % args.save_video)
            logger.info('fps@%f' % fps)
            size = [int(654 * (ori_h / ori_w)), 654]
            h = int(654 * (ori_h / ori_w))
            time_n = time.time()
            while True:
                _, image = cap.read()
                img = np.array(cv2.resize(image, (654, h)))
                cv2.imshow('raw', img)
                img_corner = np.array(cv2.resize(image, (360, int(360*(ori_h/ori_w)))))
                img = img[np.newaxis, :]
                peaks, heatmap, vectormap = sess.run([tensor_peaks, hm_up, cpm_up],
                                                     feed_dict={raw_img: img, img_size: size})
                bodys = PoseEstimator.estimate_paf(peaks[0], heatmap[0], vectormap[0])
                image = TfPoseEstimator.draw_humans(image, bodys, imgcopy=False)
                fps = round(1 / (time.time() - time_n), 2)
                image = cv2.putText(image, str(fps)+'fps', (10, 15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255))
                time_n = time.time()
                if args.video is not None:
                    image[27:img_corner.shape[0]+27, :img_corner.shape[1]] = img_corner  # [3:-10, :]
                cv2.imshow(' ', image)
                if args.save_video is not None:
                    video_saver.write(image)
                cv2.waitKey(1)
        else:
            image = common.read_imgfile(args.image)
            size = [image.shape[0], image.shape[1]]
            if image is None:
                logger.error('Image can not be read, path=%s' % args.image)
                sys.exit(-1)        
            h = int(654 * (size[0] / size[1]))
            img = np.array(cv2.resize(image, (654, h)))
            cv2.imshow('ini', img)
            img = img[np.newaxis, :]
            peaks, heatmap, vectormap = sess.run([tensor_peaks, hm_up, cpm_up], feed_dict={raw_img: img, img_size: size})
            cv2.imshow('in', vectormap[0, :, :, 0])
            bodys = PoseEstimator.estimate_paf(peaks[0], heatmap[0], vectormap[0])
            image = TfPoseEstimator.draw_humans(image, bodys, imgcopy=False)
            cv2.imshow(' ', image)
            cv2.waitKey(0)
