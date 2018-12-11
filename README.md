# unofficial-implement-for-openpose
An unofficial implement for openpose using TensorFlow.

Only basic python is used, so the code is easy to understand.

You can check the graph, internal outputs of every stage and histogram of every layer in tensorboard.

Original Repo(Caffe) : https://github.com/CMU-Perceptual-Computing-Lab/openpose

The Dataloader and Post-processing code is from [tf-pose-estimation](https://github.com/ildoonet/tf-pose-estimation)

Python 3.6

## Training
1. Download vgg19 weights file [here](http://download.tensorflow.org/models/vgg_19_2016_08_28.tar.gz) or 链接: https://pan.baidu.com/s/1ZxWKVPe4hrEhDxOpjLiUKA 提取码: widj and uzip to 'checkpoints/vgg/' (please create the path yourself)
2. Download COCO2017: 2017 Train images, 2017 Val images and 2017 Train/Val annotations [here](http://cocodataset.org/#download)
   COCO/
     -images/
       train2017/
       val2017/
     -annotations/

3. Specify the '--dathapath' and '--imgpath' in train.py to your own'COCO/annotations/' and 'COCO/images/'
3. run train.py and install requirements follow the error and run again.