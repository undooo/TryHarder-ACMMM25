
import re
import glob
import numpy as np
import os.path as osp

from dataset.base_image_dataset import BaseImageDataset

class PersonX(BaseImageDataset):

    def __init__(self, root='', verbose=True, pid_begin = 0, **kwargs):
        super(PersonX, self).__init__()
        self.dataset_dir = '/home/zhaoyujian/Dataset/PersonX_v1'
        self.train_dir = osp.join(self.dataset_dir,'*', 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir,'*' ,'query')
        self.gallery_dir = osp.join(self.dataset_dir,'*', 'bounding_box_test')

        # self._check_before_run()
        self.pid_begin = pid_begin
        train, num_train_pids = self._process_dir_train(self.dataset_dir, relabel=True)
        query = []
        gallery = []

        
            
        

        self.train = train
        self.query = query
        self.gallery = gallery

    
        if verbose:
            print("=> LTCC loaded")
            print("Dataset statistics:")
            print("  ----------------------------------------")
            print("  subset   | # ids | # images | # clothes")
            print("  ----------------------------------------")
            print("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, len(train), 0))
            print("  ----------------------------------------")
            print("  ----------------------------------------")
        self.num_train_pids = num_train_pids
    
    def _process_dir_train(self, dir_path, relabel=True):
        img_paths = glob.glob(osp.join(self.train_dir, '*.*g'))
        img_paths += (glob.glob(osp.join(self.gallery_dir, '*.*g')))
        img_paths += (glob.glob(osp.join(self.query_dir, '*.*g')))

        print(len(img_paths))

        pattern = re.compile(r'([-\d]+)_c(\d)')
        
        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
            
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        if relabel == True:
            self.pid2label = pid2label
            
        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        if relabel:
            self.pid2label = pid2label
        return dataset, len(pid_container)
