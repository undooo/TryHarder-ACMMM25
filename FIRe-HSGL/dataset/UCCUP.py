import re
import glob
import numpy as np
import os.path as osp
import os

from dataset.base_image_dataset import BaseImageDataset


class UCCUP(BaseImageDataset):
    
    def __init__(self, dataset_root='data', dataset_filename='UCCUP', verbose=True, **kwargs):
        self.dataset_dir = '/home/zhaoyujian/Dataset/UCCUP/final_low_res2'
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'gallery')

        train, num_train_pids, num_train_imgs, num_train_clothes, pid2clothes = \
            self._process_dir_train(self.train_dir)
            
        query, gallery, num_test_pids, num_query_imgs, num_gallery_imgs, num_test_clothes = \
            self._process_dir_test(self.query_dir, self.gallery_dir)
        num_total_pids = num_train_pids + num_test_pids
        num_total_imgs = num_train_imgs + num_query_imgs + num_gallery_imgs
        # num_test_imgs = num_query_imgs + num_gallery_imgs
        num_total_clothes = num_train_clothes + num_test_clothes

        if verbose:
            print("=> uccup loaded")
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
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_train_clothes = num_train_clothes
        self.pid2clothes = pid2clothes


    def _process_dir_train(self, dir_path):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        img_paths.sort()
        

        pid_container = set()
        clothes_container = set()
        for img_path in img_paths:
            basename = osp.basename(img_path)
            pid = int(basename.split('_')[0])
            clothes_id = int(basename.split('_')[1].split('cloth')[1])
            pid_container.add(pid)
            clothes_container.add(clothes_id)
        pid_container = sorted(list(pid_container))
        clothes_container = sorted(list(clothes_container))
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        clothes2label = {clothes_id: label for label, clothes_id in enumerate(clothes_container)}
        print(pid2label)
        num_pids = len(pid_container)
        num_clothes = len(clothes_container)

        dataset = []
        pid2clothes = np.zeros((num_pids, num_clothes))
        for img_path in img_paths:
            basename = osp.basename(img_path)
            pid = int(basename.split('_')[0])
            camid = int(basename.split('_C')[1].split('_')[0])
            
            clothes_id = int(basename.split('cloth')[1].split('_C')[0])
            camid -= 1  # index starts from 0
            pid = pid2label[pid]
            clothes_id = clothes2label[clothes_id]
            dataset.append((img_path, pid, clothes_id, camid))
            pid2clothes[pid, clothes_id] = 1
        
        num_imgs = len(dataset)

        return dataset, num_pids, num_imgs, num_clothes, pid2clothes
    
    def _process_dir_test(self, query_path, gallery_path):
        query_img_paths = glob.glob(osp.join(query_path, '*.jpg'))
        gallery_img_paths = glob.glob(osp.join(gallery_path, '*.jpg'))
        query_img_paths.sort()
        gallery_img_paths.sort()
        
        pid_container = set()
        clothes_container = set()
        
        for img_path in query_img_paths:
            basename = osp.basename(img_path)
            pid = int(basename.split('_')[0])
            clothes_id = int(basename.split('cloth')[1].split('_C')[0])
            pid_container.add(pid)
            clothes_container.add(clothes_id)
        for img_path in gallery_img_paths:
            basename = osp.basename(img_path)
            pid = int(basename.split('_')[0])
            clothes_id = int(basename.split('cloth')[1].split('_C')[0])
            pid_container.add(pid)
            clothes_container.add(clothes_id)
        
        pid_container = sorted(pid_container)
        clothes_container = sorted(clothes_container)
        pid2lable = {pid: label for label, pid in enumerate(pid_container)}
        clothes2label = {clothes_id: label for label, clothes_id in enumerate(clothes_container)}
        
        num_pids = len(pid_container)
        num_clothes = len(clothes_container)
        
        query_dataset = []
        gallery_dataset = []
        
        for img_path in query_img_paths:
            basename = osp.basename(img_path)
            pid = int(basename.split('_')[0])
            clothes_id = int(basename.split('cloth')[1].split('_C')[0])
            camid = int(basename.split('_C')[1].split('_')[0])
            camid -= 1
            clothes_id = clothes2label[clothes_id]
            query_dataset.append((img_path, pid, clothes_id, camid))
        for img_path in gallery_img_paths:
            basename = osp.basename(img_path)
            pid = int(basename.split('_')[0])
            clothes_id = int(basename.split('cloth')[1].split('_C')[0])
            camid = int(basename.split('_C')[1].split('_')[0])
            camid -= 1
            clothes_id = clothes2label[clothes_id]
            gallery_dataset.append((img_path, pid, clothes_id, camid))
            
        num_imgs_query = len(query_dataset)
        num_imgs_gallery = len(gallery_dataset)
        
        return query_dataset, gallery_dataset, num_pids, num_imgs_query, num_imgs_gallery, num_clothes