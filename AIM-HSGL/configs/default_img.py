import os
import yaml
from yacs.config import CfgNode as CN
import time


_C = CN()
# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Root path for dataset directory
_C.DATA.ROOT = 'DATA_ROOT'
# Dataset for evaluation
_C.DATA.DATASET = 'ltcc'
# Workers for dataloader
_C.DATA.NUM_WORKERS = 8
# Height of input image
_C.DATA.HEIGHT = 384
# Width of input image
_C.DATA.WIDTH = 192
# Batch size for training
_C.DATA.TRAIN_BATCH = 32
# Batch size for testing
_C.DATA.TEST_BATCH = 128
# The number of instances per identity for training sampler
_C.DATA.NUM_INSTANCES = 8
# extra data path
_C.DATA.EXTRA_DATA_PATH = ''


# -----------------------------------------------------------------------------
# Augmentation settings
# -----------------------------------------------------------------------------
_C.AUG = CN()
# Random crop prob
_C.AUG.RC_PROB = 0.5
# Random erase prob
_C.AUG.RE_PROB = 0.5
# Random flip prob
_C.AUG.RF_PROB = 0.5
# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Model name
_C.MODEL.NAME = 'resnet50'
# The stride for laery4 in resnet
_C.MODEL.RES4_STRIDE = 1
# feature dim
_C.MODEL.FEATURE_DIM = 4096
# Model path for resuming
_C.MODEL.RESUME = ''
# Global pooling after the backbone
_C.MODEL.POOLING = CN()
# Choose in ['avg', 'max', 'gem', 'maxavg']
_C.MODEL.POOLING.NAME = 'maxavg'
# Initialized power for GeM pooling
_C.MODEL.POOLING.P = 3
# -----------------------------------------------------------------------------
# Losses for training 
# -----------------------------------------------------------------------------
_C.LOSS = CN()
# Classification loss
_C.LOSS.CLA_LOSS = 'crossentropylabelsmooth'
# Clothes classification loss
_C.LOSS.CLOTHES_CLA_LOSS = 'cosface'
# Scale for classification loss
_C.LOSS.CLA_S = 16.
# Margin for classification loss
_C.LOSS.CLA_M = 0.
# Clothes-based adversarial loss
_C.LOSS.CAL = 'cal'
# Epsilon for clothes-based adversarial loss
_C.LOSS.EPSILON = 0.1
# Momentum for clothes-based adversarial loss with memory bank
_C.LOSS.MOMENTUM = 0.

# Pairwise loss
_C.LOSS.PAIR_LOSS = 'triplet'
# The weight for pairwise loss
_C.LOSS.PAIR_LOSS_WEIGHT = 0.0
# Scale for pairwise loss
_C.LOSS.PAIR_S = 16.
# Margin for pairwise loss
_C.LOSS.PAIR_M = 0.0
_C.LOSS.USE_TRIPLET_HD = False
_C.LOSS.LABEL_ADJUST_FACTOR = 1.0
# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.MAX_EPOCH = 60
# Start epoch for clothes classification
_C.TRAIN.START_EPOCH_CC = 25
# Start epoch for adversarial training
_C.TRAIN.START_EPOCH_ADV = 25
# Start epoch for debias
_C.TRAIN.START_EPOCH_GENERAL = 25
# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'adam'
# Learning rate
_C.TRAIN.OPTIMIZER.LR = 0.00035
_C.TRAIN.OPTIMIZER.WEIGHT_DECAY = 5e-4
# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
# Stepsize to decay learning rate
_C.TRAIN.LR_SCHEDULER.STEPSIZE = [20, 40]
# LR decay rate, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1
_C.TRAIN.AMP = False


# --------------------------------------------
# ---------------------------------
# Testing settings
# -----------------------------------------------------------------------------
_C.TEST = CN()
# Perform evaluation after every N epochs (set to -1 to test after training)
_C.TEST.EVAL_STEP = 1
# Start to evaluate after specific epoch
_C.TEST.START_EVAL = 0
# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
# Fixed random seed
_C.SEED = 1
# Perform evaluation only
_C.EVAL_MODE = False
# GPU device ids for CUDA_VISIBLE_DEVICES
_C.GPU = '0'
# Path to output folder, overwritten by command line argument
_C.OUTPUT = 'OUTPUT_PATH'
# Tag of experiment, overwritten by command line argument
_C.TAG = 'eval'
# -----------------------------------------------------------------------------
# Hyperparameters
_C.k_cal = 1.0
_C.k_kl = 1.0
# -----------------------------------------------------------------------------

def update_config(config, args):
    config.defrost()
    config.merge_from_file(args.cfg)

    if args.root:
        config.DATA.ROOT = args.root
    if args.output:
        config.OUTPUT = args.output

    if args.resume:
        config.MODEL.RESUME = args.resume
    if args.eval:
        config.EVAL_MODE = True
    
    if args.tag:
        config.TAG = args.tag

    if args.dataset:
        config.DATA.DATASET = args.dataset
    if args.gpu:
        config.GPU = args.gpu
    if args.amp:
        config.TRAIN.AMP = True

    if args.bs:
        config.DATA.TRAIN_BATCH=args.bs

    if args.triplet_w:
        config.LOSS.PAIR_LOSS_WEIGHT=args.triplet_w

    if args.arctype:
        config.TRAIN.ARCTYPE = args.arctype
    
    if args.extra_data_path:
        config.DATA.EXTRA_DATA_PATH = args.extra_data_path

    if args.use_triplet_hd == "True":
        config.LOSS.USE_TRIPLET_HD = args.use_triplet_hd

    if args.label_adjust_factor:
        config.LOSS.LABEL_ADJUST_FACTOR = args.label_adjust_factor
    # output folder
    config.OUTPUT = os.path.join(config.OUTPUT, config.DATA.DATASET, "bs"+str(config.DATA.TRAIN_BATCH)+"_"+config.TRAIN.ARCTYPE+"_tripW"+str(config.LOSS.PAIR_LOSS_WEIGHT)+"_useHD-"+str(config.LOSS.LABEL_ADJUST_FACTOR))

    config.freeze()


def get_img_config(args):
    """Get a yacs CfgNode object with default values."""
    config = _C.clone()
    update_config(config, args)

    return config
