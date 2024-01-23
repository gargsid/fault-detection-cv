import torch
import torch.nn as nn 
from torchvision.io import read_image

from torch.utils.data import Dataset
import os, sys, random
import numpy as np


class iPhoneDataset(Dataset):
    def __init__(self, dataroot, transforms, mode='train'):
        self.mode = mode # train, val, test
        self.transforms = transforms

        self.dataroot = dataroot

        if mode == 'train':
            defective_root = os.path.join(dataroot, 'Defective/train')
            non_defective_root = os.path.join(dataroot, 'Non_Defective/train')
        else:
            defective_root = os.path.join(dataroot, 'Defective/test')
            non_defective_root = os.path.join(dataroot, 'Non_Defective/test')

        self.defective_paths = [os.path.join(defective_root, image_name) for image_name in os.listdir(defective_root)]
        self.non_defective_paths = [os.path.join(non_defective_root, image_name) for image_name in os.listdir(non_defective_root)]

        self.files = self.defective_paths + self.non_defective_paths
        self.labels = [1 for _ in range(len(self.defective_paths))] + [0 for _ in range(len(self.non_defective_paths))]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image = self.transforms(read_image(self.files[idx]))
        label = self.labels[idx]
        return image, label
