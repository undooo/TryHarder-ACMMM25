import glob
import re

import os.path as osp

from .bases import BaseImageDataset
from collections import defaultdict
import pickle
class LAST(BaseImageDataset):
    

    def __init__(self, root='', verbose=True, pid_begin = 0, **kwargs):
        super(LAST, self).__init__()
        self.dataset_dir = '/zhaoyujian/Dataset/CC-ReID/last'
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.val_dir = osp.join(self.dataset_dir, 'val')
        self.gallery_dir = osp.join(self.dataset_dir, 'test')

        # self._check_before_run()
        self.pid_begin = pid_begin
        train = self._process_dir(self.train_dir, relable=True)
        query, gallery = [], []
        # query, gallery = self._process_test_dir(self.gallery_dir, relable=False)

        if verbose:
            print("=> last loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relable=False):
        img_paths = glob.glob(osp.join(dir_path, '*/*.jpg'))
        img_paths.sort()
        dataset = []
        pid_container = set()
        for ii, img_path in enumerate(img_paths):
            names = osp.basename(img_path).split('.')[0].split('_')
            clothes = names[0] + '_' + names[-1]
            pid = int(names[0])
            pid_container.add(pid)
            camid = int( ii)
            dataset.append((img_path, self.pid_begin + pid, camid, 1))
        
        return dataset
