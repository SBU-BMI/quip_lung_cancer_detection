#!/bin/bash -f

# Variables
DEFAULT_OBJ=20
DEFAULT_MPP=0.5
CANCER_TYPE=lung
MONGODB_HOST=xxx
MONGODB_PORT=27017
HEATMAP_VERSION=cancer-lung-6c_heatmap

# Base directory
export BASE_DIR=/root/quip_lung_cancer_detection
export OUT_DIR=${BASE_DIR}/data

# The username you want to download heatmaps from
#USERNAME=xxx

# The list of case_ids you want to download heaetmaps from
export CASE_LIST=${BASE_DIR}/data/raw_marking_to_download_case_list/case_list.txt

# Paths of data, log, input, and output
export JSON_OUTPUT_FOLDER=${OUT_DIR}/heatmap_jsons
export HEATMAP_TXT_OUTPUT_FOLDER=${OUT_DIR}/heatmap_txt
export LOG_OUTPUT_FOLDER=${OUT_DIR}/log
export SVS_INPUT_PATH=${OUT_DIR}/svs
export PATCH_PATH=${OUT_DIR}/patches

# other folders
export PATCH_SAMPLING_LIST_PATH=${OUT_DIR}/patch_sample_list
export RAW_MARKINGS_PATH=${OUT_DIR}/raw_marking_xy
export MODIFIED_HEATMAPS_PATH=${OUT_DIR}/modified_heatmaps
export TUMOR_HEATMAPS_PATH=${OUT_DIR}/tumor_labeled_heatmaps
export TUMOR_GROUND_TRUTH=${OUT_DIR}/tumor_ground_truth_maps
export TUMOR_IMAGES_TO_EXTRACT=${OUT_DIR}/tumor_images_to_extract
export GRAYSCALE_HEATMAPS_PATH=${OUT_DIR}/grayscale_heatmaps
export THRESHOLDED_HEATMAPS_PATH=${OUT_DIR}/thresholded_heatmaps
export PATCH_FROM_HEATMAP_PATH=${OUT_DIR}/patches_from_heatmap
export THRESHOLD_LIST=${OUT_DIR}/threshold_list/threshold_list.txt

export CAE_TRAINING_DATA=${BASE_DIR}/data/training_data_cae
export CAE_TRAINING_DEVICE=gpu0
export CAE_MODEL_PATH=${BASE_DIR}/data/models_cae
export LYM_CNN_TRAINING_DATA=${BASE_DIR}/data/training_data_cnn
export LYM_CNN_TRAINING_DEVICE=1
export LYM_CNN_PRED_DEVICE=0
export LYM_NECRO_CNN_MODEL_PATH=${BASE_DIR}/data/models_cnn
export NEC_CNN_TRAINING_DATA=${BASE_DIR}/data/training_data_cnn
export NEC_CNN_TRAINING_DEVICE=1
export NEC_CNN_PRED_DEVICE=0
export EXTERNAL_LYM_MODEL=0

if [[ -z "${CUDA_VISIBLE_DEVICES}" ]]; then
	export LYM_CNN_TRAINING_DEVICE=0
	export LYM_CNN_PRED_DEVICE=0
else
	export LYM_CNN_TRAINING_DEVICE=${CUDA_VISIBLE_DEVICES}
	export LYM_CNN_PRED_DEVICE=${CUDA_VISIBLE_DEVICES}
fi

