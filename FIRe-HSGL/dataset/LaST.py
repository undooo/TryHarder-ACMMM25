import glob
import numpy as np
import os.path as osp


class LaST(object):
    """ LaST

    Reference:
        Shu et al. Large-Scale Spatio-Temporal Person Re-identification: Algorithm and Benchmark. arXiv:2105.15076, 2021.

    URL: https://github.com/shuxjweb/last

    Note that LaST does not provide the clothes label for val and test set.
    """
    def __init__(self, dataset_root='data', dataset_filename='last', verbose=True,extra_data_path=None, **kwargs):
        self.dataset_dir = osp.join(dataset_root, dataset_filename)
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.val_query_dir = osp.join(self.dataset_dir, 'val', 'query')
        self.val_gallery_dir = osp.join(self.dataset_dir, 'val', 'gallery')
        self.test_query_dir = osp.join(self.dataset_dir, 'test', 'query')
        self.test_gallery_dir = osp.join(self.dataset_dir, 'test', 'gallery')
        self.extra_data_dir = extra_data_path
        self._check_before_run()

        pid2label, clothes2label, pid2clothes = self.get_pid2label_and_clothes2label(self.train_dir, self.extra_data_dir)

        train, num_train_pids = self._process_dir(self.train_dir, pid2label=pid2label, clothes2label=clothes2label, relabel=True, extra_path=self.extra_data_dir)
        val_query, num_val_query_pids = self._process_dir(self.val_query_dir, relabel=False)
        val_gallery, num_val_gallery_pids = self._process_dir(self.val_gallery_dir, relabel=False, recam=len(val_query))
        test_query, num_test_query_pids = self._process_dir(self.test_query_dir, relabel=False)
        test_gallery, num_test_gallery_pids = self._process_dir(self.test_gallery_dir, relabel=False, recam=len(test_query))

        num_total_pids = num_train_pids+num_val_gallery_pids+num_test_gallery_pids
        num_total_imgs = len(train) + len(val_query) + len(val_gallery) + len(test_query) + len(test_gallery)
        print("self.extra_data_path", self.extra_data_dir)
        print("=> LaST loaded")
        print("Dataset statistics:")
        print("  --------------------------------------------")
        print("  subset        | # ids | # images | # clothes")
        print("  ----------------------------------------")
        print("  train         | {:5d} | {:8d} | {:9d}".format(num_train_pids, len(train), len(clothes2label)))
        print("  query(val)    | {:5d} | {:8d} |".format(num_val_query_pids, len(val_query)))
        print("  gallery(val)  | {:5d} | {:8d} |".format(num_val_gallery_pids, len(val_gallery)))
        print("  query         | {:5d} | {:8d} |".format(num_test_query_pids, len(test_query)))
        print("  gallery       | {:5d} | {:8d} |".format(num_test_gallery_pids, len(test_gallery)))
        print("  --------------------------------------------")
        print("  total         | {:5d} | {:8d} | ".format(num_total_pids, num_total_imgs))
        print("  --------------------------------------------")

        self.train = train
        self.val_query = val_query
        self.val_gallery = val_gallery
        self.query = test_query
        self.gallery = test_gallery

        self.num_train_pids = num_train_pids
        self.num_train_clothes = len(clothes2label)
        self.pid2clothes = pid2clothes

    def get_pid2label_and_clothes2label(self, dir_path, extra_data_dir=None):
        img_paths = glob.glob(osp.join(dir_path, '*/*.jpg'))            # [103367,]
        img_paths.sort()

        pid_container = set()
        clothes_container = set()
        for img_path in img_paths:
            names = osp.basename(img_path).split('.')[0].split('_')
            clothes = names[0] + '_' + names[-1]
            pid = int(names[0])
            pid_container.add(pid)
            clothes_container.add(clothes)
       
        if extra_data_dir is not None:
            extra_img_paths = glob.glob(osp.join(self.extra_data_dir, '*.png')) + glob.glob(osp.join(self.extra_data_dir, '*.jpg'))
            extra_img_paths.sort()
            for img_path in extra_img_paths:
                names = osp.basename(img_path).split('.')[0].split('_')
                clothes = names[-1]
                pid = int(names[0])
                pid_container.add(pid)
                clothes_container.add(clothes)

        pid_container = sorted(pid_container)
        clothes_container = sorted(clothes_container)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        clothes2label = {clothes:label for label, clothes in enumerate(clothes_container)}

        num_pids = len(pid_container)
        num_clothes = len(clothes_container)

        pid2clothes = np.zeros((num_pids, num_clothes))
        for img_path in img_paths:
            names = osp.basename(img_path).split('.')[0].split('_')
            clothes = names[0] + '_' + names[-1]
            pid = int(names[0])
            pid = pid2label[pid]
            clothes_id = clothes2label[clothes]
            pid2clothes[pid, clothes_id] = 1

        return pid2label, clothes2label, pid2clothes

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.val_query_dir):
            raise RuntimeError("'{}' is not available".format(self.val_query_dir))
        if not osp.exists(self.val_gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.val_gallery_dir))
        if not osp.exists(self.test_query_dir):
            raise RuntimeError("'{}' is not available".format(self.test_query_dir))
        if not osp.exists(self.test_gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.test_gallery_dir))

    def _process_dir(self, dir_path, pid2label=None, clothes2label=None, relabel=False, recam=0, extra_path=None):
        if 'query' in dir_path:
            img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        else:
            img_paths = glob.glob(osp.join(dir_path, '*/*.jpg'))
        img_paths.sort()
        
        dataset = []
        pid_container = set()

        if extra_path is not None:
            if not osp.exists(extra_path):
                raise RuntimeError("'{}' is not available".format(extra_path))
            if not osp.isdir(extra_path):
                raise RuntimeError("'{}' is not a directory".format(extra_path))
            extra_img_paths = glob.glob(osp.join(extra_path, '*.png')) + glob.glob(osp.join(extra_path, '*.jpg'))
            extra_img_paths.sort()

            for img_path in extra_img_paths:

                names = osp.basename(img_path).split('.')[0].split('_')
                clothes = names[-1]
                pid = int(names[0])
                pid_container.add(pid)
                camid = int(recam + len(dataset))
                if relabel and pid2label is not None:
                    pid = pid2label[pid]
                if relabel and clothes2label is not None:
                    clothes_id = clothes2label[clothes]
                else:
                    clothes_id = pid
                dataset.append((img_path, pid, clothes_id, camid))


        for ii, img_path in enumerate(img_paths):
            names = osp.basename(img_path).split('.')[0].split('_')
            clothes = names[0] + '_' + names[-1]
            pid = int(names[0])
            pid_container.add(pid)
            camid = int(recam + ii)
            if relabel and pid2label is not None:
                pid = pid2label[pid]
            if relabel and clothes2label is not None:
                clothes_id = clothes2label[clothes]
            else:
                clothes_id = pid
            dataset.append((img_path, pid, clothes_id, camid))
        num_pids = len(pid_container)

            

        return dataset, num_pids