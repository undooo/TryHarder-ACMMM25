#!/bin/bash

DATASET="ltcc"
TRAIN_BATCH="32"
TRIP_WEIGHT=1.0 # Weight of the triplet loss function
HMETHOD=1.0 # Distance Adjustment Factor mentioned in the paper(1.0 indicate no use of HSDA) 


#RESUME_PATH="/path/to/pretrained/model/weight.pth"

# training with original dataset
# EXTRA_DATA_TYPE="normal_firetrip"
# EXTRA_DATA_PATH=None
# TRI_START=35


# training with extra data
# EXTRA_DATA_TYPE="5-N-cc-firetrip"
# EXTRA_DATA_PATH="/root/autodl-tmp/LTCC_ReID/5_N_cc"
# TRI_START=35

# CHPSG pretrain
# USE_INPAINT="True"
# EXTRA_DATA_TYPE="inpaint"
# EXTRA_DATA_PATH="/path/to/CHPSG/data"
# TRI_START=5
# SAVE_PER_EPOCH=1


# 定义通用的设备ID
DEVICE_ID="0"
DATASET_ROOT="/path/to/dataset/root/"
MAX_EPOCH=15
MODEL_NAME="hdetector"
OUTPUT_PATH=/root/autodl-tmp/output-CSSC/${DATASET}/${TRAIN_BATCH}_${EXTRA_DATA_TYPE}_w${TRIP_WEIGHT}_h${HMETHOD}_start${TRI_START}




python main.py \
    --gpu_devices ${DEVICE_ID} \
    --dataset ${DATASET}  \
    --dataset_root ${DATASET_ROOT}\
    --dataset_filename ${DATASET} \
    --train_batch ${TRAIN_BATCH} \
    --max_epoch ${MAX_EPOCH} \
    --HMETHOD ${HMETHOD} \
    --extra_data_path ${EXTRA_DATA_PATH}\
    --trip_weight ${TRIP_WEIGHT}\
    --tri_start_epoch ${TRI_START}\
    --save_checkpoint_itr ${SAVE_PER_EPOCH}\
    --save_dir ${OUTPUT_PATH} \
    --use_inpaint ${USE_INPAINT}\
    --save_checkpoint \


