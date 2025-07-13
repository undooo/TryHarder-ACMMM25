"""
@author:  Qizao Wang
@contact: qzwang22@m.fudan.edu.cn

TIFS 2024 paper: Exploring Fine-Grained Representation and Recomposition for Cloth-Changing Person Re-Identification
URL: https://ieeexplore.ieee.org/document/10557733
GitHub: https://github.com/QizaoWang/FIRe-CCReID
"""

from __future__ import absolute_import

import time
import datetime
import numpy as np
import os.path as osp

import torch
from torch import nn

from utils.arguments import get_args, set_log, print_args, set_gpu
from utils.util import set_random_seed
from dataset import dataset_manager, PRCC
from data_process import dataset_loader_cc, dataset_loader, dataset_loader_hardlable
from losses.triplet_loss import TripletLoss
from losses.triplet_loss_hAdap import TripletHAdapLoss
from losses.cross_entropy_loss import CrossEntropyLabelSmooth
from scheduler.warm_up_multi_step_lr import WarmupMultiStepLR
from utils.util import load_checkpoint, save_checkpoint
from model import fire, fireHD
import train_fire, test_cc, test


def main():
    args = get_args()
    set_log(args)
    print_args(args)
    use_gpu = set_gpu(args)
    set_random_seed(args.seed, use_gpu)

    print("Initializing dataset {}".format(args.dataset))
    if args.dataset == 'prcc':
        dataset = PRCC.PRCC(root=args.dataset_root, extra_data_path=args.extra_data_path)
        train_loader, query_sc_loader, query_cc_loader, gallery_loader = \
            dataset_loader_cc.get_prcc_dataset_loader(dataset, args=args, use_gpu=use_gpu)
    elif args.dataset in ['ltcc', 'deepchange', 'last','vcclothes', 'uccup']:
        dataset = dataset_manager.get_dataset(args)
        train_loader, query_loader, gallery_loader = \
            dataset_loader_cc.get_cc_dataset_loader(dataset, args=args, use_gpu=use_gpu)
    elif args.dataset in ['ccup_hard_easy']:
        dataset = dataset_manager.get_dataset(args)
        train_loader, query_loader, gallery_loader = \
            dataset_loader.get_ccup_hard_easy_dataset_loader(dataset, args=args, use_gpu=use_gpu)
    elif args.dataset in ['ccup_hardlable']:
        dataset = dataset_manager.get_dataset(args)
        train_loader, query_loader, gallery_loader = \
            dataset_loader_hardlable.get_ccup_hardlable_dataset_loader(dataset, args=args, use_gpu=use_gpu)
    else:
        dataset = dataset_manager.get_dataset(args)
        train_loader, query_loader, gallery_loader = \
            dataset_loader.get_dataset_loader(dataset, args=args, use_gpu=use_gpu)

    num_classes = dataset.num_train_pids
    
    model = fire.FIRe(pool_type='maxavg', last_stride=1, pretrain=True, num_classes=num_classes)
    classifier = fire.Classifier(feature_dim=model.feature_dim, num_classes=num_classes)

    class_criterion = CrossEntropyLabelSmooth(num_classes=num_classes, epsilon=0.1, use_gpu=use_gpu)
    if args.model_name in ['hdetector']:
        metric_criterion = TripletHAdapLoss(margin=args.margin, cfg=args)
    else: 
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

    # only test
    if args.evaluate:
        print("Evaluate only")
        if args.dataset == 'prcc':
            test_cc.test_for_prcc(args, query_sc_loader, query_cc_loader, gallery_loader, model, use_gpu, ranks=[1, 5, 10], epoch=None)
        elif args.dataset == 'ltcc':
            test_cc.test_for_ltcc(args, query_loader, gallery_loader, model, use_gpu, ranks=[1, 5, 10], epoch=None)
        elif args.dataset in ['deepchange', 'last', 'vcclothes', 'uccup']:
            test_cc.test_for_cc(args, query_loader, gallery_loader, model, use_gpu, ranks=[1, 5, 10], epoch=None)
        else:
            test.test(args, query_loader, gallery_loader, model, use_gpu, ranks=[1, 5, 10], epoch=None)
        return 0

    # train
    print("==> Start training")
    start_time = time.time()
    train_time = 0
    best_mAP, best_rank1 = -np.inf, -np.inf
    best_epoch_mAP, best_epoch_rank1 = 0, 0

    flag = False
    best_mAP_2, best_rank1_2 = -np.inf, -np.inf
    best_epoch_mAP_2, best_epoch_rank1_2 = 0, 0

    for epoch in range(start_epoch, args.max_epoch):
        start_train_time = time.time()
        train_fire.train(args, epoch + 1, dataset, train_loader, model, classifier,
                         optimizer, scheduler, class_criterion, metric_criterion, FFM_criterion, use_gpu)
        train_time += round(time.time() - start_train_time)

        if  not args.resume and (epoch + 1) % args.save_epoch == 0 and args.save_checkpoint:
            model_state_dict = model.module.state_dict() if use_gpu else model.state_dict()
            classifier_state_dict = classifier.module.state_dict() if use_gpu else classifier.state_dict()
            optimizer_state_dict = optimizer.state_dict()
            save_checkpoint({
                'model_state_dict': model_state_dict,
                'classifier_state_dict': classifier_state_dict,
                'optimizer_state_dict': optimizer_state_dict,
                'rank1': 0,
                'mAP': 0,
                'epoch': epoch,
            }, False, False, osp.join(args.save_dir, 'checkpoint_ep' + str(epoch + 1) + '.pth'))

        
        # evaluate
        if (epoch + 1) > args.start_eval_epoch and args.eval_epoch > 0 and (epoch + 1) % args.eval_epoch == 0 \
                or (epoch + 1) == args.max_epoch:
            print("==> Test")
            if args.dataset == 'prcc':
                rank1, mAP = test_cc.test_for_prcc(args, query_sc_loader, query_cc_loader,
                                                   gallery_loader, model, use_gpu, ranks=[1, 5, 10], epoch=epoch + 1)
            elif args.dataset == 'ltcc':
                rank1, mAP = test_cc.test_for_ltcc(args, query_loader, gallery_loader, model,
                                                   use_gpu, ranks=[1, 5, 10], epoch=epoch + 1)
            elif args.dataset in ['deepchange', 'last', 'vcclothes', 'uccup']:
                rank1, mAP = test_cc.test_for_cc(args, query_loader, gallery_loader, model,
                                                 use_gpu, ranks=[1, 5, 10], epoch=epoch + 1)
            else:
                rank1, mAP = test.test(args, query_loader, gallery_loader, model,
                                       use_gpu, ranks=[1, 5, 10], epoch=epoch + 1)
            if isinstance(rank1, list):
                rank1, rank1_2 = rank1
                mAP, mAP_2 = mAP
                flag = True

            is_best_mAP = mAP > best_mAP
            is_best_rank1 = rank1 > best_rank1
            if is_best_mAP:
                best_mAP = mAP
                best_epoch_mAP = epoch + 1
            if is_best_rank1:
                best_rank1 = rank1
                best_epoch_rank1 = epoch + 1

            if flag:
                is_best_mAP_2 = mAP_2 > best_mAP_2
                is_best_rank1_2 = rank1_2 > best_rank1_2
                if is_best_mAP_2:
                    best_mAP_2 = mAP_2
                    best_epoch_mAP_2 = epoch + 1
                if is_best_rank1_2:
                    best_rank1_2 = rank1_2
                    best_epoch_rank1_2 = epoch + 1
        print("==> Best mAP {:.4%}, achieved at epoch {}".format(best_mAP, best_epoch_mAP))
        print("==> Best Rank-1 {:.4%}, achieved at epoch {}".format(best_rank1, best_epoch_rank1))
        if flag:
            print("==> Best mAP_2 {:.4%}, achieved at epoch {}".format(best_mAP_2, best_epoch_mAP_2))
            print("==> Best Rank-1_2 {:.4%}, achieved at epoch {}".format(best_rank1_2, best_epoch_rank1_2))
    # if args.save_checkpoint and args.resume:
    #             model_state_dict = model.module.state_dict() if use_gpu else model.state_dict()
    #             classifier_state_dict = classifier.module.state_dict() if use_gpu else classifier.state_dict()
    #             optimizer_state_dict = optimizer.state_dict()
    #             save_checkpoint({
    #                 'model_state_dict': model_state_dict,
    #                 'classifier_state_dict': classifier_state_dict,
    #                 'optimizer_state_dict': optimizer_state_dict,
    #                 'rank1': rank1,
    #                 'mAP': mAP,
    #                 'epoch': epoch,
    #             }, is_best_mAP, is_best_rank1, osp.join(args.save_dir, 'checkpoint_ep' + str(epoch + 1) +
    #                                                     '_mAP_' + str(round(mAP * 100, 2)) + '_rank1_' + str(
    #                 round(rank1 * 100, 2)) + '.pth'))
    
        

    print("==> Best mAP {:.4%}, achieved at epoch {}".format(best_mAP, best_epoch_mAP))
    print("==> Best Rank-1 {:.4%}, achieved at epoch {}".format(best_rank1, best_epoch_rank1))
    with open('./record.txt', 'a') as f:
        if args.resume:
            s = 'dataset: ' + args.dataset + ', ' + 'model: ' +  args.resume + ', '  \
                + 'rank1: ' + str(best_rank1)  + ', ' +  'mAP: ' + str(best_mAP)  + '\n'
        else:
            s = 'dataset: ' + args.dataset + ', ' + 'model: ' + 'none' + ', '  \
                + 'rank1: ' + str(best_rank1)  + ', ' +  'mAP: ' + str(best_mAP)  + '\n'
        f.write(s)
    if flag:
        print("==> Best mAP_2 {:.4%}, achieved at epoch {}".format(best_mAP_2, best_epoch_mAP_2))
        print("==> Best Rank-1_2 {:.4%}, achieved at epoch {}".format(best_rank1_2, best_epoch_rank1_2))
        with open('./record.txt', 'a') as f:
            if args.resume:
                s = 'dataset: ' + args.dataset + ', ' + 'model: ' +  args.resume + ', '  \
                    + 'rank2: ' + str(best_rank1_2)  + ', ' +  'mAP: ' + str(best_mAP_2) + '\n'
            else:
                s = 'dataset: ' + args.dataset + ', ' + 'model: ' + 'none' + ', '  \
                    + 'rank2: ' + str(best_rank1_2)  + ', ' +  'mAP: ' + str(best_mAP_2) + '\n'
            f.write(s)

    # time using info
    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    train_time = str(datetime.timedelta(seconds=train_time))
    print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))
    
    with open('./record.txt', 'a') as f:
        s = 'dataset: ' + args.dataset + ', ' + 'model: ' +  args.resume + ', '  \
            + 'rank1: ' + str(best_rank1)  + ', ' +  'mAP: ' + str(best_mAP) 


if __name__ == '__main__':
    main()