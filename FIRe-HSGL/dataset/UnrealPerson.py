import glob
import re

import os.path as osp
import os

from collections import defaultdict
import pickle

from dataset.base_image_dataset import BaseImageDataset


class UnrealPerson(BaseImageDataset):
    
    def __init__(self, dataset_root='data', dataset_filename='LTCC_ReID', verbose=True, **kwargs):
        self.dataset_dir = '/home/zhaoyujian/Dataset/UnrealPerson'
        self.dataset_dir = [osp.join(self.dataset_dir, d) for d in os.listdir(self.dataset_dir)]
        self.train_dir = [osp.join(d,'images') for d in self.dataset_dir]

        train, num_train_pids, num_train_imgs, num_train_clothes, pid2clothes = \
            self._process_dir_train(self.train_dir)
            
        query, gallery, num_test_pids, num_query_imgs, num_gallery_imgs, num_test_clothes = \
            [], [], 0, 0, 0, 0
        num_total_pids = num_train_pids + num_test_pids
        num_total_imgs = num_train_imgs + num_query_imgs + num_gallery_imgs
        # num_test_imgs = num_query_imgs + num_gallery_imgs
        num_total_clothes = num_train_clothes + num_test_clothes

        if verbose:
            print("=> UnrealPerson loaded")
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


    def _process_dir_train(self, dir_path, relabel=True):
        cid_container = set()
        pid_container = set()
        pid_container_sep = defaultdict(set)
        img_paths =[]
        for d in dir_path:
            if not os.path.exists(d):
                assert False, 'Check unreal data dir'

            iii = glob.glob(osp.join(d, '*.*g'))
            print(d,len(iii))
            img_paths.extend( iii )
        pattern = re.compile(r'unreal_v([\d]+).([\d]+)/images/([-\d]+)_c([\d]+)_([\d]+)')
        for img_path in img_paths:
            sid,pv, pid, cid,fid = map(int, pattern.search(img_path).groups())
            cid_container.add((sid,cid))
            pid_container_sep[pv].add((pv,pid))
            pid_container.add((pv,pid))
        for k in pid_container_sep.keys():
            print("Unreal pids ({}): {}".format(k,len(pid_container_sep[k])))
        print("Unreal cams: {}".format(len(cid_container)))
        
        pid2label = {pid:label for label, pid in enumerate(pid_container)}
        cid2label = {cid:label for label, cid in enumerate(cid_container)}
        
        if relabel == True:
            self.pid2label = pid2label
            self.cid2label = cid2label
            
        
        dataset = []
        for img_path in img_paths:
            sid,pv, pid, cid,fid = map(int, pattern.search(img_path).groups())
            if (pv,pid) not in pid_container:continue
            if (sid,cid) not in cid_container:continue
            if relabel: 
                pid = pid2label[(pv,pid)]
                camid = cid2label[(sid,cid)]
                dataset.append((img_path, pid, camid))
                
        return dataset, len(pid_container), len(dataset), 0, None
