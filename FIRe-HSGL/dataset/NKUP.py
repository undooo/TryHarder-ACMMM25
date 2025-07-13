import re
import glob
import numpy as np
import os.path as osp

from dataset.base_image_dataset import BaseImageDataset


class NKUP(BaseImageDataset):

    def __init__(self, dataset_root='data', dataset_filename='LTCC_ReID', verbose=True, **kwargs):
        self.dataset_dir = '/home/zhaoyujian/Dataset/NKUP'
        self.train_dir     = osp.join(self.dataset_dir, 'bounding_box_train')
        self.gallery_dir   = osp.join(self.dataset_dir, 'bounding_box_test')
        self.query_dir     = osp.join(self.dataset_dir, 'query')

        train, num_train_pids, num_train_imgs, num_train_clothes, pid2clothes = \
            self._process_dir_train(self.train_dir)
        query, num_query_pids, num_query_imgs = self._process_dir_query(self.query_dir)
        gallery, num_gallery_pids, num_gallery_imgs = self._process_dir_gallery(self.gallery_dir)
        
        # query, gallery, num_test_pids, num_query_imgs, num_gallery_imgs, num_test_clothes = \
            # self._process_dir_test(self.query_dir, self.gallery_dir)
        # num_total_pids = num_train_pids + num_test_pids
        num_total_imgs = num_train_imgs + num_query_imgs + num_gallery_imgs
        # num_test_imgs = num_query_imgs + num_gallery_imgs
        num_total_clothes = 0
        if verbose:
            print("=> NKUP loaded")
            print("Dataset statistics:")
            print("  ----------------------------------------")
            print("  subset   | # ids | # images | # clothes")
            print("  ----------------------------------------")
            print("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_clothes))
            # print("  test     | {:5d} | {:8d} | {:9d}".format(num_test_pids, num_test_imgs, num_test_clothes))
            print("  query    | {:5d} | {:8d} |".format(num_query_pids, num_query_imgs))
            print("  gallery  | {:5d} | {:8d} |".format(num_gallery_pids, num_gallery_imgs))
            print("  ----------------------------------------")
            # print("  total    | {:5d} | {:8d} | {:9d}".format(num_total_pids, num_total_imgs, num_total_clothes))
            print("  ----------------------------------------")

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_train_clothes = num_train_clothes
        self.pid2clothes = pid2clothes

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

    def _process_dir_train(self, dir_path, relabel=True):
        img_paths = glob.glob(osp.join(dir_path, '*'))
        img_paths.sort()

        pid_container = set()
        camid_container = set()
        for img_path in img_paths:
            names = osp.basename(img_path).split('_')
            pid = int(names[0])
            pid_container.add(pid)
            camid = int(names[2][1:3])
            camid_container.add(camid)
        pid_container = sorted(list(pid_container))
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        num_pids = len(pid_container)

        dataset = []
        for img_path in img_paths:
            names = osp.basename(img_path).split('_')
            pid = int(names[0])
            if relabel:
                pid = pid2label[pid]
            camid = int(names[2][1:3])
            camid -= 1
            dataset.append((img_path, pid, camid))
            
        num_imgs = len(dataset)

        return dataset, num_pids, num_imgs, 0, None

    def _process_dir_query(self, dir_path, relabel=False):
        query_img_paths =  glob.glob(osp.join(dir_path, '*'))
        query_img_paths.sort()
        pid_container = set()
        for img_path in query_img_paths:
            names = osp.basename(img_path).split('_')
            pid = int(names[0])
            pid_container.add(pid)
            camid = int(names[2][1:3])
            pid_container.add(pid)
            
        pid_container = sorted(pid_container)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        num_pids = len(pid_container)
        query_dataset = []
        for img_path in query_img_paths:
            names = osp.basename(img_path).split('_')
            pid = int(names[0])
            camid = int(names[2][1:3])
            camid -= 1
            query_dataset.append((img_path, pid, camid))
        num_imgs_query = len(query_dataset)
        return query_dataset, num_pids, num_imgs_query
    
    def _process_dir_gallery(self, dir_path, relabel=False):
        gallery_img_paths = glob.glob(osp.join(dir_path, '*'))
        gallery_img_paths.sort()
        pid_container = set()
        for img_path in gallery_img_paths:
            names = osp.basename(img_path).split('_')
            pid = int(names[0])
            pid_container.add(pid)
            camid = int(names[2][1:3])
            pid_container.add(pid)
        pid_container = sorted(pid_container)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        num_pids = len(pid_container)
        gallery_dataset = []
        for img_path in gallery_img_paths:
            names = osp.basename(img_path).split('_')
            pid = int(names[0])
            camid = int(names[2][1:3])
            camid -= 1
            gallery_dataset.append((img_path, pid, camid))
        num_imgs_gallery = len(gallery_dataset)
        return gallery_dataset, num_pids, num_imgs_gallery
    
    def _process_dir_test(self, query_path, gallery_path):
        query_img_paths =  glob.glob(osp.join(query_path, '*'))
        gallery_img_paths = glob.glob(osp.join(gallery_path, '*'))
        query_img_paths.sort()
        gallery_img_paths.sort()

        pid_container = set()
        for img_path in query_img_paths:
            names = osp.basename(img_path).split('_')
            pid = int(names[0])
            pid_container.add(pid)
            camid = int(names[2][1:3])
            pid_container.add(pid)
        for img_path in gallery_img_paths:
            names = osp.basename(img_path).split('_')
            pid = int(names[0])
            pid_container.add(pid)
            camid = int(names[2][1:3])
            pid_container.add(pid)
        pid_container = sorted(pid_container)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        num_pids = len(pid_container)

        query_dataset = []
        gallery_dataset = []
        for img_path in query_img_paths:
            names = osp.basename(img_path).split('_')
            pid = int(names[0])
            camid = int(names[2][1:3])
            camid -= 1
            query_dataset.append((img_path, pid, camid))
        for img_path in gallery_img_paths:
            names = osp.basename(img_path).split('_')
            pid = int(names[0])
            camid = int(names[2][1:3])
            camid -= 1
            gallery_dataset.append((img_path, pid, camid))
        
        num_imgs_query = len(query_dataset)
        num_imgs_gallery = len(gallery_dataset)

        return query_dataset, gallery_dataset, num_pids, num_imgs_query, num_imgs_gallery, 0
