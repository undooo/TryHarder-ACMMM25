# prcc
DATASET="prcc"
TRIPLET_W=0.1 # Weight of the triplet loss function
USE_TRIPLET_HD="True" # Whether use HSDA strategy
LABEL_ADJUST_FACTOR=1.05 # Distance Adjustment Factor mentioned in the paper 
BS=64

# Train only with raw data
ARCTYPE="normal"
EXTRA_DATA_PATH=None

# Train with extra data
# ARCTYPE="5_N_cc"
# EXTRA_DATA_PATH="/path/to/PRCC_ReID/SHSG/Data"

python -m torch.distributed.launch --nproc_per_node=1 --master_port 12357 \
        main.py\
        --dataset ${DATASET} \
        --cfg configs/res50_cels_cal.yaml \
        --root /root/autodl-tmp\
        --output /root/autodl-tmp/output-cal-sensitive\
        --triplet_w ${TRIPLET_W}\
        --arctype ${ARCTYPE} \
        --extra_data_path ${EXTRA_DATA_PATH} \
        --bs ${BS}\
        --use_triplet_hd ${USE_TRIPLET_HD}\
        --label_adjust_factor ${LABEL_ADJUST_FACTOR}\
        --gpu 0 
