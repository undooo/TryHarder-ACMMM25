"""
@author:  Qizao Wang
@contact: qzwang22@m.fudan.edu.cn

TIFS 2024 paper: Exploring Fine-Grained Representation and Recomposition for Cloth-Changing Person Re-Identification
URL: https://ieeexplore.ieee.org/document/10557733
GitHub: https://github.com/QizaoWang/FIRe-CCReID
"""

from __future__ import absolute_import

import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import cv2
import time
import datetime
import numpy as np
import os.path as osp
import tqdm
import torch
from torch import nn

from utils.arguments import get_args, set_log, print_args, set_gpu
from utils.util import set_random_seed
from dataset import dataset_manager, PRCC
from data_process import dataset_loader_cc, dataset_loader
from losses.triplet_loss import TripletLoss
from losses.cross_entropy_loss import CrossEntropyLabelSmooth
from scheduler.warm_up_multi_step_lr import WarmupMultiStepLR
from utils.util import load_checkpoint, save_checkpoint
from model import fire
import train_fire, test_cc, test


def main():
    args = get_args()
    set_log(args)
    print_args(args)
    use_gpu = set_gpu(args)
    set_random_seed(args.seed, use_gpu)

    print("Initializing dataset {}".format(args.dataset))
    if args.dataset == 'prcc':
        dataset = PRCC.PRCC(dataset_root=args.dataset_root, dataset_filename=args.dataset_filename)
        train_loader, query_sc_loader, query_cc_loader, gallery_loader = \
            dataset_loader_cc.get_prcc_dataset_loader(dataset, args=args, use_gpu=use_gpu)
    elif args.dataset in ['ltcc', 'deepchange', 'last','vcclothes']:
        dataset = dataset_manager.get_dataset(args)
        train_loader, query_loader, gallery_loader = \
            dataset_loader_cc.get_cc_dataset_loader(dataset, args=args, use_gpu=use_gpu)
    else:
        dataset = dataset_manager.get_dataset(args)
        train_loader, query_loader, gallery_loader = \
            dataset_loader.get_dataset_loader(dataset, args=args, use_gpu=use_gpu)

    num_classes = dataset.num_train_pids
    model = fire.FIRe(pool_type='maxavg', last_stride=1, pretrain=True, num_classes=num_classes)
    classifier = fire.Classifier(feature_dim=model.feature_dim, num_classes=num_classes)

    class_criterion = CrossEntropyLabelSmooth(num_classes=num_classes, epsilon=0.1, use_gpu=use_gpu)
    metric_criterion = TripletLoss(margin=args.margin)
    FFM_criterion = fire.AttrAwareLoss(scale=args.temperature, epsilon=args.epsilon)

    parameters = list(model.parameters()) + list(classifier.parameters())
    optimizer = torch.optim.Adam(params=[{'params': parameters, 'initial_lr': args.lr}],
                                 lr=args.lr, weight_decay=args.weight_decay)

    start_epoch = args.start_epoch  # 0 by default
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        # model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        model.load_pretrain_model(args.resume, num_classes=num_classes)
        print('==> Loaded model from {}'.format(args.resume))
        classifier = fire.Classifier(feature_dim=model.feature_dim, num_classes=num_classes)
        # classifier.load_state_dict(checkpoint['classifier_state_dict'], strict=True)
        print('==> Loaded classifier from {}'.format(args.resume))

        if 'optimizer_state_dict' in checkpoint.keys():
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if use_gpu:
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.cuda()
        start_epoch = checkpoint['epoch'] + 1  # start from the next epoch

    scheduler = WarmupMultiStepLR(optimizer, milestones=args.step_milestones, gamma=args.gamma,
                                  warmup_factor=args.warm_up_factor, last_epoch=start_epoch - 1)

    if use_gpu:
        model = nn.DataParallel(model).cuda()
        classifier = nn.DataParallel(classifier).cuda()

    
    target_layers = [model.module.backbone[7][-1]]
    
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    for (img_dir, pid, clothes_id, camid) in tqdm.tqdm(dataset.query_cloth_changed):
        # Prepare image
        img_path = img_dir
        # img = Image.open(img_path).convert('RGB')
        # img = np.array(img, dtype=np.uint8)
        img = cv2.imread(img_path)
        img_tensor = data_transform(img)
        input_tensor = torch.unsqueeze(img_tensor, dim=0)
        cam = GradCAM(model=model, target_layers=target_layers)
        # targets = [ClassifierOutputTarget(281)]     # cat
        targets = None # dog

        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(img.astype(dtype=np.float32)/255.,grayscale_cam, use_rgb=True)
        # visualization = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
        cv2.imwrite('./vis_picture/' + str(pid) + '_' + img_dir.split('/')[-1], visualization)



if __name__ == '__main__':
    main()