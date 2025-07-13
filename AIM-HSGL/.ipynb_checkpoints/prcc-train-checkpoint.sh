# ltcc
DATASET="prcc"
TRIPLET_W=0.1
USE_TRIPLET_HD="True"
LABEL_ADJUST_FACTOR=1.05
BS=64

# 原版
ARCTYPE="normal"
EXTRA_DATA_PATH=None

# ## 每个ID仅挑选五张图片、仅换衣
# ARCTYPE="5_N_cc"
# EXTRA_DATA_PATH="/root/autodl-tmp/prcc/rgb/5_N_cc"

## 每个ID仅挑选一张图片、换衣+多角度
# 自动生成带时间戳的日志文件名
# TIMESTAMP=$(date +"%Y%m%d-%H%M%S")
# LOG_FILE="/root/autodl-tmp/output/log/ltcc-train-${TIMESTAMP}.log"

python -m torch.distributed.launch --nproc_per_node=1 --master_port 12358 \
        main.py\
        --dataset ${DATASET} \
        --cfg configs/res50_cels_cal.yaml \
        --root /root/autodl-tmp\
        --output /root/autodl-tmp/output-aim\
        --triplet_w ${TRIPLET_W}\
        --arctype ${ARCTYPE} \
        --extra_data_path ${EXTRA_DATA_PATH} \
        --bs ${BS}\
        --use_triplet_hd ${USE_TRIPLET_HD}\
        --label_adjust_factor ${LABEL_ADJUST_FACTOR}\
        --gpu 0 
