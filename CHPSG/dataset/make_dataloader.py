import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from .bases import ImageDataset
from PIL import Image
from timm.data.random_erasing import RandomErasing
from .prcc import PRCC
from .last import LAST
import torch.distributed as dist
import os

__factory = {
    'prcc':PRCC,
    'last':LAST,
}

def make_dataloader():
    train_transforms = T.Compose([
            T.Resize((256, 128), interpolation=3),
            T.ToTensor(),
            # T.Normalize(mean=[0.5], std=[0.5]),
        ])

    mask_transforms = T.Compose([
            T.Resize((256, 128), interpolation=3),
            T.ToTensor(),
            # T.Normalize(mean=[0.5], std=[0.5]),
        ])

    num_workers = 4
    dataset = __factory['last'](root='')
    train_set = ImageDataset(dataset.train, train_transforms, mask_transforms)
    train_loader = DataLoader(
        train_set, batch_size=64, shuffle=True, num_workers=num_workers)

    train_loader.num_classes = dataset.num_train_pids
    return train_loader

    
