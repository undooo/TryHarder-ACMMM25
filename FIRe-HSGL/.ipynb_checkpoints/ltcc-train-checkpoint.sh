#!/bin/bash

DATASET="ltcc"
TRAIN_BATCH="32"

# 普通训练
# EXTRA_DATA_TYPE="normal"
# EXTRA_DATA_PATH=None
# TRIP_WEIGHT=1.0
# HMETHOD=1.0

# 额外数据训练
TRIP_WEIGHT=1.1
EXTRA_DATA_TYPE="5-N-cc"
EXTRA_DATA_PATH="/root/autodl-tmp/LTCC_ReID/5_N_cc"
HMETHOD=1.05


# 定义通用的设备ID
DEVICE_ID="0"
DATASET_ROOT="/root/autodl-tmp"
MAX_EPOCH="100"
MODEL_NAME="hdetector"
SAVE_EPOCH_PERIOD=10
FG_START_EPOCH=1
ITER_LOG_PERIOD=100
# 打印当前执行的命令（可选）
# 执行Python脚本
python main.py \
    --gpu_devices ${DEVICE_ID} \
    --dataset ${DATASET}  \
    --dataset_root ${DATASET_ROOT}\
    --extra_data_path ${EXTRA_DATA_PATH}\
    --dataset_filename ${DATASET} \
    --train_batch ${TRAIN_BATCH} \
    --max_epoch ${MAX_EPOCH} \
    --trip_weight ${TRIP_WEIGHT}\
    --save_dir /root/autodl-tmp/output-fire/${DATASET}/${TRAIN_BATCH}_${EXTRA_DATA_TYPE}_w${TRIP_WEIGHT}_h${HMETHOD} \
    --HMETHOD ${HMETHOD} \
    --model_name ${MODEL_NAME} \
    --save_epoch ${SAVE_EPOCH_PERIOD} \
    --fg_start_epoch ${FG_START_EPOCH} \
    --save_checkpoint \
    --iter_log_period   ${ITER_LOG_PERIOD}


