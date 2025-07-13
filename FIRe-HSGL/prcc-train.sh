#!/bin/bash

DATASET="prcc"
TRAIN_BATCH="32"
TRIP_WEIGHT=1.0 # Weight of the triplet loss function
HMETHOD=1.0 # Distance Adjustment Factor mentioned in the paper(1.0 indicate no use of HSDA) 


#RESUME="/path/to/pretrained/model/weight.pth"

# training with original dataset
EXTRA_DATA_TYPE="normal"
EXTRA_DATA_PATH=None


# training with extra data
# EXTRA_DATA_TYPE="5-N-cc"
# EXTRA_DATA_PATH="/path/to/PRCC_ReID/SHSG/Data"




# 
DEVICE_ID="0"
DATASET_ROOT="/path/to/dataset/root/"
MAX_EPOCH="60"
MODEL_NAME="hdetector"
SAVE_EPOCH_PERIOD=10
FG_START_EPOCH=1
ITER_LOG_PERIOD=100
OUTPUT_PATH=/path/to/output-fire/${DATASET}/${TRAIN_BATCH}_${EXTRA_DATA_TYPE}_w${TRIP_WEIGHT}_h${HMETHOD}



python main.py \
    --gpu_devices ${DEVICE_ID} \
    --resume ${RESUME}\
    --dataset ${DATASET}  \
    --dataset_root ${DATASET_ROOT}\
    --extra_data_path ${EXTRA_DATA_PATH}\
    --dataset_filename ${DATASET} \
    --train_batch ${TRAIN_BATCH} \
    --max_epoch ${MAX_EPOCH} \
    --trip_weight ${TRIP_WEIGHT}\
    --save_dir ${OUTPUT_PATH} \
    --HMETHOD ${HMETHOD} \
    --model_name ${MODEL_NAME} \
    --save_epoch ${SAVE_EPOCH_PERIOD} \
    --fg_start_epoch ${FG_START_EPOCH} \
    --save_checkpoint \
    --iter_log_period   ${ITER_LOG_PERIOD}


