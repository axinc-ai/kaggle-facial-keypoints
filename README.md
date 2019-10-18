# kaggle-facial-keypoints
This is a RES-Net based Neural Network model that is challenging a kaggle contest, [Facial Keypoints Detection](https://www.kaggle.com/c/facial-keypoints-detection).


## Description
The objective of this task is to predict keypoint positions on face images. Datasets are available on the [kaggle website](https://www.kaggle.com/c/facial-keypoints-detection/data).

## Demo
This is an example of my results.
[result_image]()

## Requirements
As a Deep Learning framework, I used [Pytorch](https://pytorch.org) version 1.3.

## Usage 
To train a model on your environment, 
```
python3 train.py 
```
If you want to use my pretrained model, 
```
python3 train.py --load
```

## TODO
- [ ] Cross Validation
- [ ] TensorboardX
- [ ] log system
- [ ] Improve usability

## Licence
[MIT]()

## Author
[sngyo](https://github.com/sngyo)
