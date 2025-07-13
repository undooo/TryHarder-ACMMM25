#!/bin/bash

DATASET="ltcc"
TRAIN_BATCH="32"


RESUME_PATH="/root/autodl-tmp/output-CSSC/ltcc/32_inpaint_w1.0_h1.0_start5/"
CP="checkpoint_ep1_mAP_12.83_rank1_33.47.pth"
# 普通训练
# EXTRA_DATA_TYPE="normal_firetrip"
# EXTRA_DATA_PATH=None
# TRIP_WEIGHT=1.0
# HMETHOD=1.05
# TRI_START=35


# 额外数据训练
TRIP_WEIGHT=1.35
EXTRA_DATA_TYPE="5-N-cc-firetrip"
EXTRA_DATA_PATH="/root/autodl-tmp/LTCC_ReID/5_N_cc"
HMETHOD=1.05
TRI_START=35
SAVE_PER_EPOCH=5

# inpaint训练
# USE_INPAINT="True"
# TRIP_WEIGHT=1.0
# EXTRA_DATA_TYPE="inpaint"
# EXTRA_DATA_PATH=None
# HMETHOD=1.0
# TRI_START=0
# SAVE_PER_EPOCH=1


# 定义通用的设备ID
DEVICE_ID="0"
DATASET_ROOT="/root/autodl-tmp"
MAX_EPOCH=120
MODEL_NAME="hdetector"




# 打印当前执行的命令（可选）
# 执行Python脚本
python main.py \
    --gpu_devices ${DEVICE_ID} \
    --dataset ${DATASET}  \
    --resume ${RESUME_PATH}${CP}\
    --dataset_root ${DATASET_ROOT}\
    --dataset_filename ${DATASET} \
    --train_batch ${TRAIN_BATCH} \
    --max_epoch ${MAX_EPOCH} \
    --HMETHOD ${HMETHOD} \
    --extra_data_path ${EXTRA_DATA_PATH}\
    --trip_weight ${TRIP_WEIGHT}\
    --tri_start_epoch ${TRI_START}\
    --save_checkpoint_itr ${SAVE_PER_EPOCH}\
    --save_dir /root/autodl-tmp/output-CSSC-inpaint/${DATASET}/${CP}/${TRAIN_BATCH}_${EXTRA_DATA_TYPE}_w${TRIP_WEIGHT}_h${HMETHOD}_start${TRI_START} \
    --save_checkpoint \


