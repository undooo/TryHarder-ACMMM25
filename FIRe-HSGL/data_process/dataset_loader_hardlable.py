"""
@author:  Qizao Wang
@contact: qzwang22@m.fudan.edu.cn

TIFS 2024 paper: Exploring Fine-Grained Representation and Recomposition for Cloth-Changing Person Re-Identification
URL: https://ieeexplore.ieee.org/document/10557733
GitHub: https://github.com/QizaoWang/FIRe-CCReID
"""

from __future__ import print_function, absolute_import

from utils.util import read_image
from data_process import samplers, transform

from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class ImageClothDataset_Hardlable(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, is_hard = self.dataset[index]
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)
        return img, pid, camid, is_hard


def get_ccup_hardlable_dataset_loader(dataset, args, use_gpu=True):
    transform_train, transform_test = transform.get_transform(args)

    sampler = samplers.RandomIdentityHardlableSampler(dataset.train, batch_size=args.train_batch,
                                             num_instances=args.num_instances)

    pin_memory = use_gpu
    train_loader = DataLoader(
        ImageClothDataset_Hardlable(dataset.train, transform=transform_train),
        sampler=sampler, batch_size=args.train_batch, num_workers=args.num_workers,
        pin_memory=pin_memory, drop_last=True,
    )

    query_loader = DataLoader(
        ImageClothDataset_Hardlable(dataset.query, transform=transform_test),
        batch_size=args.test_batch, shuffle=False, num_workers=args.num_workers,
        pin_memory=pin_memory, drop_last=False,
    )

    gallery_loader = DataLoader(
        ImageClothDataset_Hardlable(dataset.gallery, transform=transform_test),
        batch_size=args.test_batch, shuffle=False, num_workers=args.num_workers,
        pin_memory=pin_memory, drop_last=False,
    )

    return train_loader, query_loader, gallery_loader
