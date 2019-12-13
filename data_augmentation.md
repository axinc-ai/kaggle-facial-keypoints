---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.3.0
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# This notebook is for creting augmented dataset using imgaug for kaggle facial feature points detection 

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

import imgaug as ia

from tqdm import tqdm
```

loading original training dataset

```python
org_training_file = "data/resized226.csv"
image_size = 226

df = pd.read_csv(org_training_file)
df['Image'] = df['Image'].apply(lambda im:np.fromstring(im, sep=' '))
```

```python
df.info()
```

```python
df.count()
```

```python
df = df.dropna()
df.count()
```

X: Image  
y: feature points (teacher information)

```python
X = np.vstack(df['Image'].values)
X = X.astype(np.float32)
X = X.reshape(X.shape[0], image_size, image_size)

y = df[df.columns[:-1]].values
```

```python
from imgaug.augmentables.kps import Keypoint, KeypointsOnImage

# get keypoints
def get_kps(y_data):
    """
    param: y_data: annotations set
    """
    kps = []
    for i in range(len(y_data) // 2):
        kps.append(Keypoint(y_data[2 * i], y_data[2 * i + 1]))
    return kps


# convert an array to keypoints
def arr_to_kps(annotations):
    kpss = []
    for anno in annotations:
        kpss.append(get_kps(anno))
    return kpss

# gray scale image to rbg image
def gray_to_rgb(imgs):
    """
    param: imgs: (batch, height, width)
    """
    rgb_imgs = np.zeros((imgs.shape[0], imgs.shape[1], imgs.shape[2], 3))
    for i in range(3):
        rgb_imgs[:, :, :, i] = imgs
    return rgb_imgs.astype('uint8')
        
# test
# kps = get_kps(y[0])
# kpsoi = KeypointsOnImage(kps, shape=X[0].shape)
# X_ = gray_to_rgb(X)
# ia.imshow(kpsoi.draw_on_image(X_[0], size=2))
```

```python
X = gray_to_rgb(X)  # original image
y = arr_to_kps(y)  # keypoints
```

## Augmentation function


Now to the actual augmentation. We want to apply an affine transformation, which will alter both the image and the keypoints. We choose a bit of translation and rotation as our transoformation. Additionally, we add a bit of color jittering to the mix. That color jitter is only going to affect the image, not the keypoints.

```python
import imgaug.augmenters as iaa
ia.seed(3)

seq = iaa.Sequential([
    iaa.Affine(translate_percent={"x": (-0.05, 0.05), "y": (-0.15, 0.15)}),
    iaa.Affine(scale=(0.8, 1.0), rotate=(-30, 30), shear=(-15, 15)),
    iaa.PiecewiseAffine(scale=(0.01, 0.05))
    # iaa.Affine(translate_px={"x":(10, 30)}, rotate=(-10, 10)),
    # iaa.AddToHueAndSaturation((-50, 50))  #color jitter, only affects the image
])
```

```python
aug_X = []
aug_y = []

AGM_RPT = 5  # 1 original image to AGM_RPT images

# It may takes few minutes
for j in range(AGM_RPT):
    for i in tqdm(range(X.shape[0])):
        aug_img, aug_kpsoi = seq(image=X[i], keypoints=y[i])
        aug_X.append(aug_img)
        aug_y.append(aug_kpsoi)

    

# img = ia.imresize_single_image(img.astype('uint8'), (96, 96))
# img_aug, kpsoi_aug = seq(image=img, keypoints=kpsoi)
```

```python
# check how transoformation worked !
id = 10

ia.imshow(
    np.hstack([
        KeypointsOnImage(y[id], shape=X[id].shape).draw_on_image(X[id], size=2),
        KeypointsOnImage(aug_y[id], shape=aug_X[id].shape).draw_on_image(aug_X[id], size=2)
    ])
)
```

```python
## transformed images to save csv file
# new_X = []
# for img in aug_X:
#     new_X.append(img[:, :, 0])
# new_X = np.array(new_X).reshape(np.array(new_X).shape[0], -1)
# print(new_X.shape)

new_X = np.array(aug_X)[:, :, :, 0].reshape(len(aug_X), -1)
print(new_X.shape)
```

```python
# transformed annotations to save csv file
new_y = []
for anno in aug_y:
    new_y.append(KeypointsOnImage(anno, shape=X[0].shape).to_xy_array().reshape(-1))
new_y = np.array(new_y)
new_y.shape
```

```python
columns = []
for col in df.columns:
    columns.append(col)
```

```python
new_y_df = pd.DataFrame(data=new_y.astype('float64'), columns=columns[:-1])
```

```python
new_X_df = pd.DataFrame([], columns=['Image'], index=range(len(new_X)))

for i in tqdm(range(len(new_X))):
    str_ = ""
    for j in range(new_X.shape[1]):
        str_ += "{} ".format(new_X[i, j])
    new_X_df.iloc[i, 0] = str_
    # new_X_df.iloc[i, 0] = np.array2string(new_X[i], separator=' ').replace("[", "").replace("]", "")
```

```python
# np.fromstring(np.array2string(new_X[0], separator=' ').replace("[", "").replace("]", ""), sep=' ')
new_X_df.head()
```

```python
y_save_filename = "data/rotate_30.csv"

new_df = pd.concat([new_y_df, new_X_df], axis=1)
new_df.to_csv(y_save_filename, index=False)
```

```python
new_df.head()
```

## Resize for imagenet  (only use once)

```python
from imgaug import augmenters as iaa

seq = iaa.Sequential([
    iaa.Resize((226, 226)),
    iaa.Affine(rotate=(-30, 30))
])
```

```python
X = gray_to_rgb(X)
y = arr_to_kps(y)
aug_X = []
aug_y = []
for i in range(X.shape[0]):
    aug_img, aug_kpsoi = seq(image=X[i], keypoints=y[i])
    aug_X.append(aug_img)
    aug_y.append(aug_kpsoi)
```

```python
ia.imshow(KeypointsOnImage(y[0], shape=X[0].shape).draw_on_image(X[0], size=2))
ia.imshow(KeypointsOnImage(aug_y[0], shape=aug_X[0].shape).draw_on_image(aug_X[0], size=2))

```

```python
new_X = []
for img in aug_X:
    new_X.append(img[:, :, 0])
new_X = np.array(new_X).reshape(np.array(new_X).shape[0], -1)
print(aug_X[0].shape)
print(new_X.shape)
```

```python
# transformed annotations to save csv file
new_y = []
for anno in aug_y:
    new_y.append(KeypointsOnImage(anno, shape=X[0].shape).to_xy_array().reshape(-1))
new_y = np.array(new_y)
new_y.shape
```

```python
new_y[0]
```

```python
columns = []
for col in df.columns:
    columns.append(col)
```

```python
new_y_df = pd.DataFrame(data=new_y.astype('float64'), columns=columns[:-1])
```

```python
new_X_df = pd.DataFrame([], columns=['Image'], index=range(len(new_X)))

for i in range(len(new_X)):
    str_ = ""
    for j in range(new_X.shape[1]):
        str_ += "{} ".format(new_X[i, j])
    new_X_df.iloc[i, 0] = str_
    # new_X_df.iloc[i, 0] = np.array2string(new_X[i], separator=' ').replace("[", "").replace("]", "")
```

```python
new_df = pd.concat([new_y_df, new_X_df], axis=1)
new_df.to_csv('data/resized226_rotate_30.csv', index=False)
```

## shifting keypoints

```python
img_pad = ia.pad(img, left=10)
kpsoi_pad = kpsoi.shift(x=10)
ia.imshow(kpsoi_pad.draw_on_image(img_pad, size=2))
```

```python
arr = kpsoi.to_xy_array()
print(arr.shape)
print(arr)
print(arr.reshape(-1))
```
