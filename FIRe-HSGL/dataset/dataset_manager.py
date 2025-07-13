from __future__ import print_function, absolute_import

from dataset.LTCC import LTCC
from dataset.CelebreID import CelebreID
from dataset.DeepChange import DeepChange
from dataset.LaST import LaST
from dataset.CCUP import CCUP
from dataset.NKUP import NKUP
from dataset.ClonedPerson import ClonedPerson
from dataset.PERSONX import PersonX
from dataset.UnrealPerson import UnrealPerson
from dataset.VCCLOTHES import VCClothes
from dataset.CelebreID_light import CelebreID_light
from dataset.UCCUP import UCCUP
from dataset.CCUP_Hardlable import CCUP_Hardlable

__img_factory = {
    'ltcc': LTCC,
    'celeb': CelebreID,
    'ccup': CCUP,
    'deepchange': DeepChange,
    'last': LaST,
    'nkup': NKUP,
    'clonedperson': ClonedPerson,
    'personx': PersonX,
    'unrealperson': UnrealPerson,
    'vcclothes': VCClothes,
    'celeb_light': CelebreID_light,
    'ccup_hard_easy': CCUP,
    'uccup': UCCUP,
    'ccup_hardlable': CCUP_Hardlable,
}


def get_dataset(args):
    name = args.dataset
    if name not in __img_factory.keys():
        raise KeyError("Invalid dataset, got '{}', but expected to be one of {}".format(name, __img_factory.keys()))
    if name in ['ltcc']:
        dataset = __img_factory[name](root=args.dataset_root, extra_path=args.extra_data_path)
    else:
        dataset = __img_factory[name](dataset_root=args.dataset_root, dataset_filename=args.dataset_filename, extra_data_path=args.extra_data_path)
    return dataset