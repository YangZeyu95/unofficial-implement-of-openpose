import tensorflow as tf
import os
import sys
from dataset import get_dataflow, batch_dataflow
from dataflow import COCODataPaths
import cv2
from tensorflow.contrib import slim
import vgg
from cpm import CpmStage1

echos = 100
batch_size = 10
checkpoint_path = 'checkpoints/train/'
vgg19_ckpt_path = "checkpoints/vgg/vgg_19.ckpt"

curr_dir = os.path.dirname(__file__)
annot_path_train = '/run/user/1000/gvfs/smb-share:server=192.168.1.2,share=data/yzy/dataset/' \
                 'Realtime_Multi-Person_Pose_Estimation-master/training/dataset/COCO/annotations/' \
                 'person_keypoints_train2017.json'
img_dir_train = '/run/user/1000/gvfs/smb-share:server=192.168.1.2,share=data/yzy/dataset/' \
              'Realtime_Multi-Person_Pose_Estimation-master/training/dataset/COCO/images/train2017/'
annot_path_val = '/run/user/1000/gvfs/smb-share:server=192.168.1.2,share=data/yzy/dataset/' \
                 'Realtime_Multi-Person_Pose_Estimation-master/training/dataset/COCO/annotations/' \
                 'person_keypoints_val2017.json'
img_dir_val = '/run/user/1000/gvfs/smb-share:server=192.168.1.2,share=data/yzy/dataset/' \
              'Realtime_Multi-Person_Pose_Estimation-master/training/dataset/COCO/images/val2017/'

coco_data_train = COCODataPaths(annot_path=annot_path_train, img_dir=img_dir_train)
coco_data_val = COCODataPaths(annot_path=annot_path_val, img_dir=img_dir_val)
df = get_dataflow(coco_data_val)
train_samples = df.size()
batch_df = batch_dataflow(df, batch_size)

with tf.name_scope('inputs'):
    raw_img = tf.placeholder(tf.float32, shape=[None, 368, 368, 3])
    mask_hm = tf.placeholder(dtype=tf.float32, shape=[None, 46, 46, 19])
    mask_cpm = tf.placeholder(dtype=tf.float32, shape=[None, 46, 46, 38])
    hm = tf.placeholder(dtype=tf.float32, shape=[None, 46, 46, 19])
    cpm = tf.placeholder(dtype=tf.float32, shape=[None, 46, 46, 38])

with slim.arg_scope(vgg.vgg_arg_scope()):
    vgg_outputs, end_points = vgg.vgg_19(raw_img)

net = CpmStage1(inputs_x=vgg_outputs, mask_cpm=mask_cpm, mask_hm=mask_hm, gt_hm=hm, gt_cpm=cpm)
hm_pre, cpm_pre, loss = net.gen_net()

global_step = tf.Variable(0, name='global_step', trainable=False)
learning_rate = tf.train.exponential_decay(1e-4, global_step, 400, 0.8, staircase=True)
tf.summary.scalar("lr", learning_rate)
with tf.name_scope('train'):
    train = tf.train.AdamOptimizer(learning_rate).minimize(loss=loss, global_step=global_step)

tf.summary.image('input', raw_img, max_outputs=2)

tf.summary.image('hm', hm[:, :, :, 0:1], max_outputs=2)
tf.summary.image('hm_pre', hm_pre[:, :, :, 0:1], max_outputs=2)

tf.summary.image('cpm', cpm[:, :, :, 0:1], max_outputs=2)
tf.summary.image('cpm_pre', cpm_pre[:, :, :, 0:1], max_outputs=2)

variables_can_be_restored = list(set(tf.get_collection_ref(tf.GraphKeys.GLOBAL_VARIABLES)).intersection(tf.train.list_variables(vgg19_ckpt_path)))
restorer = tf.train.Saver(variables_can_be_restored)
merged = tf.summary.merge_all()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
saver = tf.train.Saver(tf.global_variables())

with tf.Session(config=config) as sess:
    writer = tf.summary.FileWriter(checkpoint_path, sess.graph)
    sess.run(tf.group(tf.global_variables_initializer(),
                      tf.local_variables_initializer()))
    restorer.restore(sess, vgg19_ckpt_path)
    for echo in range(echos):
        for i, data in enumerate(batch_df):
            # sess.run(train, feed_dict={raw_img: data[0],
            #                            mask_cpm: data[1],
            #                            mask_hm: data[2],
            #                            cpm: data[3],
            #                            hm: data[4]})
            # if i == 2:
            for p in range(1000):
                total_loss, _, summary = sess.run([loss, train, merged], feed_dict={raw_img: data[0],
                                                                                    mask_cpm: data[1],
                                                                                    mask_hm: data[2],
                                                                                    cpm: data[3],
                                                                                    hm: data[4]})
                print(total_loss)
                writer.add_summary(summary, echo * len(batch_df) + i + p)
            # cv2.imshow("j", data[0][0][1][:][:][:])
        # cv2.waitKey(0)