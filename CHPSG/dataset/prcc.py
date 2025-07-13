import glob
import re

import os.path as osp

from .bases import BaseImageDataset
from collections import defaultdict
import pickle
class PRCC(BaseImageDataset):
    """
    Market1501
    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.
    URL: http://www.liangzheng.org/Project/project_reid.html

    Dataset statistics:
    # identities: 1501 (+1 for background)
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """
    

    def __init__(self, root='', verbose=True, pid_begin = 0, **kwargs):
        super(PRCC, self).__init__()
        self.dataset_dir = '/zhaoyujian/Dataset/CC-ReID/PRCC/rgb'
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.val_dir = osp.join(self.dataset_dir, 'val')
        self.gallery_dir = osp.join(self.dataset_dir, 'test')

        # self._check_before_run()
        self.pid_begin = pid_begin
        train = self._process_dir(self.train_dir, relable=True)
        query, gallery = self._process_test_dir(self.gallery_dir, relable=False)

        if verbose:
            print("=> prcc loaded")
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
        
        pid_dirs_path = glob.glob(osp.join(dir_path, '*'))
        print(len(pid_dirs_path))
        dataset = []
        pid_container = set()
        camid_mapper = {'A':1, 'B':2, 'C':3}
        
        for pid_dir_path in pid_dirs_path:
            img_paths = glob.glob(osp.join(pid_dir_path, '*.jp*'))
            for img_path in img_paths:
                pid = int(osp.basename(pid_dir_path))
                pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        
        for pid_dir_path in pid_dirs_path:
            img_paths = glob.glob(osp.join(pid_dir_path, '*.jp*'))
            for img_path in img_paths:
                pid = int(osp.basename(pid_dir_path))
                camid = camid_mapper[osp.basename(img_path)[0]]
                camid -= 1
                if relable: pid = pid2label[pid]
                dataset.append((img_path, self.pid_begin + pid, camid, 1))
        
        return dataset

    def _process_test_dir(self, dir_path, relable=False):
        camid_dirs_path = glob.glob(osp.join(dir_path, '*'))
        
        query = []
        gallery = []
        pid_container = set()
        camid_mapper = {'A':1, 'B':2, 'C':3}
        
        for camid_dir_path in camid_dirs_path:
            if 'enrich' in camid_dir_path:
                continue
            pid_dir_paths = glob.glob(osp.join(camid_dir_path, '*'))
            for pid_dir_path in pid_dir_paths:
                # print(pid_dir_path)
                pid = int(osp.basename(pid_dir_path))
                img_paths = glob.glob(osp.join(pid_dir_path, '*'))
                for img_path in img_paths:
                    camid = camid_mapper[osp.basename(camid_dir_path)[0]]
                    camid -= 1
                    if camid == 0:
                        query.append((img_path, self.pid_begin + pid, camid, 1))
                    else:
                        if camid == 2:
                            gallery.append((img_path, self.pid_begin + pid, camid, 1))
                        
                    
        return query, gallery