import re
import glob
import numpy as np
import os.path as osp
from tqdm import tqdm

from dataset.base_image_dataset import BaseImageDataset


class ClonedPerson(BaseImageDataset):
    
    def __init__(self, dataset_root='data', dataset_filename='LTCC_ReID', verbose=True, **kwargs):
        self.dataset_dir = '/home/zhaoyujian/Dataset/ClonedPerson'
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.query_dir = osp.join(self.dataset_dir, 'test/query')
        self.gallery_dir = osp.join(self.dataset_dir, 'test/gallery')

        train, num_train_pids, num_train_imgs, num_train_clothes, pid2clothes = \
            self._process_dir_train(self.dataset_dir)
            
        query, gallery, num_test_pids, num_query_imgs, num_gallery_imgs, num_test_clothes = \
            [], [], 0, 0, 0, 0
        num_total_pids = num_train_pids + num_test_pids
        num_total_imgs = num_train_imgs + num_query_imgs + num_gallery_imgs
        # num_test_imgs = num_query_imgs + num_gallery_imgs
        num_total_clothes = num_train_clothes + num_test_clothes

        if verbose:
            print("=> clonedperson loaded")
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
        img_paths = glob.glob(osp.join(self.train_dir, '*.*g'))
        img_paths += (glob.glob(osp.join(self.gallery_dir, '*.*g')))
        img_paths += (glob.glob(osp.join(self.query_dir, '*.*g')))
        img_paths.sort()
        
        pattern = re.compile(r'([-\d]+)_s([-\d]+)_c([-\d]+)_f([-\d]+)')

        data = []
        all_pids = {}
        camera_offset = [0, 0, 0, 4, 4, 8, 12, 12, 12, 12, 16, 16, 20]
        fps = 24

        for fpath in tqdm(img_paths):
            fname = osp.basename(fpath)  # filename: id6_s2_c2_f6.jpg
            pid, scene, cam, frame = map(int, pattern.search(fname).groups())
            scene = 2
            if pid == -1: continue
            if pid not in all_pids:
                all_pids[pid] = len(all_pids)
            if relabel:
                pid = all_pids[pid]
            camid = camera_offset[scene] + cam  # make it starting from 0
            time = frame / fps 
            data.append((fpath, pid, camid))

        return data, len(all_pids), len(data), 0, None
