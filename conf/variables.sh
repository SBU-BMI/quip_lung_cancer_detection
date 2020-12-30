#!/bin/bash

# Variables
DEFAULT_OBJ=20
DEFAULT_MPP=0.5
CANCER_TYPE=quip
MONGODB_HOST=quip4.bmi.stonybrook.edu
MONGODB_PORT=27017
HEATMAP_VERSION=cancer-lung-6c_20200210_hanle

# Base directory
BASE_DIR=/root/quip_lung_cancer_detection
OUT_DIR=${BASE_DIR}/data

# The username you want to download heatmaps from
#USERNAME=xxx

# The list of case_ids you want to download heaetmaps from
CASE_LIST=${BASE_DIR}/data/raw_marking_to_download_case_list/case_list.txt

# Paths of data, log, input, and output
JSON_OUTPUT_FOLDER=${OUT_DIR}/heatmap_jsons
HEATMAP_TXT_OUTPUT_FOLDER=${OUT_DIR}/heatmap_txt
LOG_OUTPUT_FOLDER=${OUT_DIR}/log
SVS_INPUT_PATH=${OUT_DIR}/svs
PATCH_PATH=${OUT_DIR}/patches

# other folders
PATCH_SAMPLING_LIST_PATH=${OUT_DIR}/data/patch_sample_list
RAW_MARKINGS_PATH=${OUT_DIR}/raw_marking_xy
MODIFIED_HEATMAPS_PATH=${OUT_DIR}/modified_heatmaps
TUMOR_HEATMAPS_PATH=${OUT_DIR}/tumor_labeled_heatmaps
TUMOR_GROUND_TRUTH=${OUT_DIR}/tumor_ground_truth_maps
TUMOR_IMAGES_TO_EXTRACT=${OUT_DIR}/tumor_images_to_extract
GRAYSCALE_HEATMAPS_PATH=${OUT_DIR}/grayscale_heatmaps
THRESHOLDED_HEATMAPS_PATH=${OUT_DIR}/thresholded_heatmaps
PATCH_FROM_HEATMAP_PATH=${OUT_DIR}/patches_from_heatmap
THRESHOLD_LIST=${OUT_DIR}/threshold_list/threshold_list.txt

CAE_TRAINING_DATA=${BASE_DIR}/data/training_data_cae
CAE_TRAINING_DEVICE=gpu0
CAE_MODEL_PATH=${BASE_DIR}/data/models_cae
LYM_CNN_TRAINING_DATA=${BASE_DIR}/data/training_data_cnn
LYM_CNN_TRAINING_DEVICE=gpu0
LYM_CNN_PRED_DEVICE=gpu0
LYM_NECRO_CNN_MODEL_PATH=${BASE_DIR}/data/models_cnn
NEC_CNN_TRAINING_DATA=${BASE_DIR}/data/training_data_cnn
NEC_CNN_TRAINING_DEVICE=gpu1
NEC_CNN_PRED_DEVICE=gpu0
EXTERNAL_LYM_MODEL=0

if [[ -z "${CUDA_VISIBLE_DEVICES}" ]]; then
	LYM_CNN_TRAINING_DEVICE=0
	LYM_CNN_PRED_DEVICE=0
else
	LYM_CNN_TRAINING_DEVICE=${CUDA_VISIBLE_DEVICES}
	LYM_CNN_PRED_DEVICE=${CUDA_VISIBLE_DEVICES}
fi

