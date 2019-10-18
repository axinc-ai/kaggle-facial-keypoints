# kaggle-facial-keypoints
This is a RES-Net based Neural Network model that is challenging a kaggle contest, [Facial Keypoints Detection](https://www.kaggle.com/c/facial-keypoints-detection).


## Description
The objective of this task is to predict keypoint positions on face images. Datasets are available on the [kaggle website](https://www.kaggle.com/c/facial-keypoints-detection/data).

## Demo
![result_image](https://github.com/axinc-ai/kaggle-facial-keypoints/blob/master/result.png)

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

## Resutls
Score : 3.96974

## TODO
- [ ] Cross Validation (looks my model tends to overfit)
- [ ] TensorboardX
- [ ] log system
- [ ] Improve usability
- [ ] Refactoring generate CSV for submitting the results
- [ ] Add comments to the functions

## Licence
[MIT](https://github.com/axinc-ai/kaggle-facial-keypoints/blob/master/LICENSE.txt)

## Author
[sngyo](https://github.com/sngyo)
