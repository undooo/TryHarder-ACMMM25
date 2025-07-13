import re
import glob
import numpy as np
import os.path as osp
import os

from dataset.base_image_dataset import BaseImageDataset


class CCUP_Hardlable(BaseImageDataset):
    
    def __init__(self, dataset_root='data', dataset_filename='LTCC_ReID', verbose=True, **kwargs):
        self.dataset_dir = '/home/zhaoyujian/Dataset/CCUP/ccup_divide'
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'gallery')

        train, num_train_pids, num_train_imgs, num_train_clothes, pid2clothes = \
            self._process_dir_train(self.dataset_dir)
            
        query, gallery, num_test_pids, num_query_imgs, num_gallery_imgs, num_test_clothes = \
            [], [], 0, 0, 0, 0
        num_total_pids = num_train_pids + num_test_pids
        num_total_imgs = num_train_imgs + num_query_imgs + num_gallery_imgs
        # num_test_imgs = num_query_imgs + num_gallery_imgs
        num_total_clothes = num_train_clothes + num_test_clothes

        if verbose:
            print("=> CCUP Hardlable loaded")
            print("Dataset statistics:")
            print("  ----------------------------------------")
            print("  subset   | # ids | # images | # clothes")
            print("  ----------------------------------------")
            print("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_clothes))
            # print("  test     | {:5d} | {:8d} | {:9d}".format(num_test_pids, num_test_imgs, num_test_clothes))
            print("  query    | {:5d} | {:8d} |".format(num_test_pids, num_query_imgs))
            print("  gallery  | {:5d} | {:8d} |".format(num_test_pids, num_gallery_imgs))
            print("  ----------------------------------------")
            print("  total    | {:5d} | {:8d} | {:9d}".format(num_total_pids, num_total_imgs, num_total_clothes))
            print("  ----------------------------------------")

        self.train = train
        self.query = []
        self.gallery = []

        self.num_train_pids = num_train_pids


    def _process_dir_train(self, dir_path):
        img_paths = []
        for dir in os.listdir(dir_path):
            print(dir)
            img_paths.extend(glob.glob(osp.join(dir_path, dir, '*.jpg')))
        
        # img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        img_paths.sort()
        
        pattern = re.compile(r'([-\d]+)_C([\d]+)_.*')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            pid_container.add(pid)
        pid_container = sorted(list(pid_container))
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        num_pids = len(pid_container)

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            camid -= 1  # index starts from 0
            pid = pid2label[pid]
            if pid > 4500 and pid < 4750:
                is_hard = 1
                #print('Hard sample',(img_path, self.pid_begin + pid, camid, 1, is_hard))

            else:
                is_hard = 0
            dataset.append((img_path, pid, camid, is_hard)) # cloth_id = 0
        
        num_imgs = len(dataset)

        return dataset, num_pids, num_imgs, 0, None
