import tensorflow as tf
import numpy as np
import os
 
 
 
#=============================================================================#
# write images and label in tfrecord file and read them out
def encode_to_tfrecords(tfrecords_filename, data_num): 
    ''' write into tfrecord file '''
    if os.path.exists(tfrecords_filename):
        os.remove(tfrecords_filename)
 
    writer = tf.python_io.TFRecordWriter('./'+tfrecords_filename) # 创建.tfrecord文件，准备写入
 
    for i in range(data_num):
        img_raw = np.random.randint(0,255,size=(56,56))
        img_raw = img_raw.tostring()
        example = tf.train.Example(features=tf.train.Features(
                feature={
                'label': tf.train.Feature(int64_list = tf.train.Int64List(value=[i])),     
                'img_raw':tf.train.Feature(bytes_list = tf.train.BytesList(value=[img_raw]))
                }))
        writer.write(example.SerializeToString()) 
 
    writer.close()
    return 0
 
def decode_from_tfrecords(filename_queue, is_batch):
 
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)   #返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })  #取出包含image和label的feature对象
    image = tf.decode_raw(features['img_raw'],tf.uint8)
    image = tf.reshape(image, [512,512,3])
    label = tf.cast(features['label'], tf.int64)
 
    if is_batch:
        batch_size = 3
        min_after_dequeue = 10
        capacity = min_after_dequeue+3*batch_size
        image, label = tf.train.shuffle_batch([image, label],
                                                          batch_size=batch_size, 
                                                          num_threads=3, 
                                                          capacity=capacity,
                                                          min_after_dequeue=min_after_dequeue)
    return image, label
 
#=============================================================================#
 
if __name__=='__main__':
    # make train.tfrecord
#     train_filename = "train.tfrecords"
#     encode_to_tfrecords(train_filename,100)
# ##    # make test.tfrecord
#     test_filename = 'test.tfrecords'
#     encode_to_tfrecords(test_filename,10)
 
#    run_test = True
    train_filename = 'dataset/train.tfrecords'
    filename_queue = tf.train.string_input_producer([train_filename], num_epochs=None) #读入流中
    train_image, train_label = decode_from_tfrecords(filename_queue, is_batch=True)
 
    # filename_queue = tf.train.string_input_producer([test_filename],num_epochs=None) #读入流中
    # test_image, test_label = decode_from_tfrecords(filename_queue, is_batch=True)
    with tf.Session() as sess: #开始一个会话
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        coord=tf.train.Coordinator()
        threads= tf.train.start_queue_runners(coord=coord)
 
        try:
            # while not coord.should_stop():
            for i in range(90):
                example, l = sess.run([train_image,train_label])#在会话中取出image和label
                print('train:')
                print(l) 
                # texample, tl = sess.run([test_image, test_label])
                # print('test:')
                # print(texample,tl)
        except tf.errors.OutOfRangeError:
            print('Done reading')
        finally:
            coord.request_stop()
 
        coord.request_stop()
        coord.join(threads)