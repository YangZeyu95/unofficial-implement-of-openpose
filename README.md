# Unofficial-Implement-of-Openpose
<p align="left">
<img src="https://github.com/YangZeyu95/unofficial-implement-of-openpose/blob/master/readme/IMG_4063.GIF", width="720">
</p>　　

You can check the full result on [YouTube](https://youtu.be/v-CC0g7whTs) or [bilibili](https://www.bilibili.com/video/av38475550/)　　

An easy implement of openpose using TensorFlow.

Only basic python is used, so the code is easy to understand.

You can check the graph, internal outputs of every stage and histogram of every layer in tensorboard.

Original Repo(Caffe) : https://github.com/CMU-Perceptual-Computing-Lab/openpose.

The Dataloader and Post-processing code is from [tf-pose-estimation](https://github.com/ildoonet/tf-pose-estimation).

Python 3.6　　
<p align="left">
<img src="https://github.com/YangZeyu95/unofficial-implement-of-openpose/blob/master/readme/graph_run%3D.png", width="720">
</p>　

## Training
1. Download vgg19 weights file [here](http://download.tensorflow.org/models/vgg_19_2016_08_28.tar.gz) or 链接: https://pan.baidu.com/s/1ZxWKVPe4hrEhDxOpjLiUKA 提取码: widj and uzip to 'checkpoints/vgg/' (please create the path yourself).
2. Download COCO2017: 2017 Train images, 2017 Val images and 2017 Train/Val annotations [here](http://cocodataset.org/#download).  
make sure have this structure:  
-COCO/  
　-images/  
　　-train2017/  
　　-val2017/  
　-annotations/    

3. Specify '--annot_path_train' and '--img_path_train' in train.py to your own 'COCO/annotations/' and 'COCO/images/'.
4. run train.py `python train.py` and install requirements follow the error and run again.
<p align="left">
<img src="https://github.com/YangZeyu95/unofficial-implement-of-openpose/blob/master/readme/loss2.svg", width="720">
</p>　　
　

## Test
Specify --checkpoint_path to the folder includes checkpoint files in run.py.　　

+ running on webcam `python run.py`　　
+ running on video `python run.py --video images/video.avi`　　
+ running on image`python run.py --image images/ski.jpg`　　

pretrained model on COCO 2017 is available [here](https://drive.google.com/drive/folders/1wQp6tU3xOyO4FF54YZShEmLuwsGLVAQA?usp=sharing) or 链接: https://pan.baidu.com/s/1FX-_YJQFwPRd0ECvVDli6Q 提取码: xwnk, this checkpoint includes fine-tuned vgg weights.　　


