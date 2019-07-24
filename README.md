## Description
This is a project fork from [EAST](https://github.com/SakuraRiven/EAST)
the origin use:
* Only RBOX part is implemented.
* Using dice loss instead of class-balanced cross-entropy loss. Some codes refer to [argman/EAST](https://github.com/argman/EAST) and [songdejia/EAST](https://github.com/songdejia/EAST)
* The pre-trained model provided achieves __82.79__ F-score on ICDAR 2015 Challenge 4 using only the 1000 images. see [here](http://rrc.cvc.uab.es/?ch=4&com=evaluation&view=method_info&task=1&m=52405) for the detailed results.

while i use giou loss function for giou-east, use efficient-east and iou loss for efficient-east, the result is as follow:
the pretrained model can be download at [here](https://pan.baidu.com/s/1_rW0SYm9ycJPAPHKZWc2ZQ) password: __9qmd__

| Model | Loss | Recall | Precision | F-score | 
| - | - | - | - | - |
| Original | CE | 72.75 | 80.46 | 76.41 |
| Re-Implement | Dice | 81.27 | 84.36 | 82.79 |
| giou-east | Giou | 78.91 | 84.65 | 81.68 |
| efficient-east | Iou | 80.21 | 82.06 | 81.12


## Prerequisites
Only tested on
* Anaconda3
* Python 3.7.1
* PyTorch 1.0.1
* Shapely 1.6.4
* opencv-python 4.0.0.21
* lanms 1.0.2

When running the script, if some module is not installed you will see a notification and installation instructions. __if you failed to install lanms, please update gcc and binutils__. The update under conda environment is:

    conda install -c omgarcia gcc-6
    conda install -c conda-forge binutils

## Installation
### 1. Clone the repo

```
git clone https://github.com/SakuraRiven/EAST.git
cd EAST
```

### 2. Data & Pre-Trained Model
* Download Train and Test Data: [ICDAR 2015 Challenge 4](http://rrc.cvc.uab.es/?ch=4&com=downloads). Cut the data into four parts: train_img, train_gt, test_img, test_gt.

* Download pre-trained VGG16 from PyTorch: [VGG16](https://drive.google.com/open?id=1HgDuFGd2q77Z6DcUlDEfBZgxeJv4tald) and our trained EAST model: [EAST](https://drive.google.com/open?id=1AFABkJgr5VtxWnmBU3XcfLJvpZkC2TAg). Make a new folder ```pths``` and put the download pths into ```pths```
  
```
mkdir pths
mv east_vgg16.pth vgg16_bn-6c64b313.pth pths/
```

Here is an example:
```
.
├── EAST
│   ├── evaluate
│   └── pths
└── ICDAR_2015
    ├── test_gt
    ├── test_img
    ├── train_gt
    └── train_img
```
## Train
Modify the parameters in ```train.py``` and run:
```
CUDA_VISIBLE_DEVICES=0,1 python train.py
```
## Detect
Modify the parameters in ```detect.py``` and run:
```
CUDA_VISIBLE_DEVICES=0 python detect.py
```
## Evaluate
* The evaluation scripts are from [ICDAR Offline evaluation](http://rrc.cvc.uab.es/?ch=4&com=mymethods&task=1) and have been modified to run successfully with Python 3.7.1.
* Change the ```evaluate/gt.zip``` if you test on other datasets.
* Modify the parameters in ```eval.py``` and run:
```
CUDA_VISIBLE_DEVICES=0 python eval.py
```
