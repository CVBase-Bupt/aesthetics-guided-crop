# aesthetics-guided-crop

A keras implementation of Auto-Cropping System framework by Hao Zhang(lichds@bupt.edu.cn).

## Performance
On FLMS:

|        |IOU     |BDE     |
| :------: | :------: | :------: |
|FLMS|0.843|0.027|

On CUHK-ICD:

|        |IOU     |BDE     |
| :------: | :------: | :------: |
|P1|0.827|0.032|
|P2|0.816|0.035|
|P3|0.805|0.036|

With the faster version:
On FLMS:

|        |IOU     |BDE     |
| :------: | :------: | :------: |
|FLMS|0.842|0.029|

On CUHK-ICD:

|        |IOU     |BDE     |
| :------: | :------: | :------: |
|P1|0.824|0.032|
|P2|0.812|0.036|
|P3|0.803|0.036|

## Requirements
python 2.7+ (or python 3)

numpy

keras 2.1.6 with tensorflow backend.

## Installation
```
git clone https://github.com/buptzh111/aesthetics-guided-crop.git 
```

### Weight File

```
https://pan.baidu.com/s/1J4pBVMKas6zaiu7AOAZWyg code: 1243
```
After download the package,  decompress it with
```
tar -xvf weights.tar
```
Subsequently, the weight files(```saliency.h5``` and ```regression.h5```) will be placed in the ```model``` directory.

## Usage

To run the demo, just run the bash file:
```
bash crop_image.sh
```
If you want to crop your own image (or image list), you can put your image file or directory into the ```example_images``` directory,
or set the ```img_path``` in  ```crop_image.sh```:
```
python -u src/crop.py [img_path]
```
For a faster version:
```
python -u src/crop_faster.py [img_path]
```
The crop result will save in ```crop_result``` directory.If you want to save at other places, just change the ```save_path``` in ```crop_image.sh```:
```
python -u src/crop.py [img_path] [save_path]
```
We also record the coordinates of cropping box and saliency box and we normalized them.The format of data is ```[x1,x2,y1,y2]```,where ```x``` denotes height, ```y```denotes width.```(x1,y1)```is the top-left coordinate of a box, and ```(x2,y2)``` is the right-down.  
