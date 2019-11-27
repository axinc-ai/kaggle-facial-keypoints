# kaggle-facial-keypoints
This is a RES-Net based Neural Network model that is challenging a kaggle contest, [Facial Keypoints Detection](https://www.kaggle.com/c/facial-keypoints-detection).


## Description
The objective of this task is to predict keypoint positions on face images. Datasets are available on the [kaggle website](https://www.kaggle.com/c/facial-keypoints-detection/data).

## Demo
![result_image](https://github.com/axinc-ai/kaggle-facial-keypoints/blob/master/result.png)

## Requirements
As a Deep Learning framework, I used [Pytorch](https://pytorch.org) version 1.3.

## Usage 
If you want to training the model on your own environment, 
```shell script
$ python3 train.py 
```
If you want to use my pretrained model to fine-tuning, `--load` option is available.  
```shell script
$ python3 train.py --load
```
When you would like to generate kaggle submission file,
```shell script
$ python3 inference.py
``` 
Then type the model file name (generally saved as `save.pt`)
```
save model file : [YOUR MODEL FILE NAME].pt
```


## Results
| Kaggle Public Score  | Training epoch|
|:-----:|:-----:|    
| 3.25420 | 1000 epochs |
|3.32088| 500 epochs|
|3.50467|300 epochs|




## TODO
- [ ] log system
- [ ] Improve usability
- [ ] Add comments to the functions
- [ ] Generate Results image function 

## Licence
[MIT](https://github.com/axinc-ai/kaggle-facial-keypoints/blob/master/LICENSE.txt)

## Author
[sngyo](https://github.com/sngyo)
