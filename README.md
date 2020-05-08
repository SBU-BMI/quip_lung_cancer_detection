# quip_lung_cancer_detection
This repo contains training and prediction code for 2-class Lung cancer detection using pretrained resnet.

# Dependencies

 - [Pytorch 0.4.0](http://pytorch.org/)
 - Torchvision 0.2.0
 - cv2 (3.4.1)
 - [Openslide 1.1.1](https://openslide.org/api/python/)
 - [sklearn](https://scikit-learn.org/stable/)
 - [PIL](https://pillow.readthedocs.io/en/3.1.x/reference/Image.html)

## Setup conf/variables.sh
- Change the BASE_DIR to the path of your folder after you clone the git repo 

## Training
- Go to folder "training_codes", run python train_lung_2classes.py

## WSIs prediction
- Go to folder "scripts", run bash svs_2_heatmap.sh

