# quip_lung_cancer_detection
This repo contains training and prediction code for 6-class Lung cancer detection using pretrained resnet.

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
- Go to folder "training_codes", run python train_lung_john_6classes.py

## WSIs prediction
- Go to folder "scripts", run bash svs_2_heatmap.sh


# Docker Instructions

Build the docker image by: 

`docker build -t cancer_prediction .`  (Note the dot at the end). 

## Step 1:
Create folder named "data" and subfolders below on the host machine:

- data/svs: to contains *.svs files
- data/patches: to contain output from patch extraction
- data/log: to contain log files
- data/heatmap_txt: to contain prediction output
- data/heatmap_jsons: to contain prediction output as json files

## Step 2:
- Run the docker container as follows: 

```
nvidia-docker run --name cancer_prediction_pipeline -itd -v <path-to-data>:/data -e CUDA_VISIBLE_DEVICES='<cuda device id>' cancer_prediction svs_2_heatmap.sh 
```

CUDA_VISIBLE_DEVICES -- set to select the GPU to use 

The following example runs the cancer detection pipeline. It will process images in /home/user/data/svs and output the results to /home/user/data. 

```
nvidia-docker run --name cancer_prediction_pipeline -itd -v /home/user/data:/data -e CUDA_VISIBLE_DEVICES='0' cancer_prediction svs_2_heatmap.sh
```
