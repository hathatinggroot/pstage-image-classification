import os
import cv2
import pandas as pd
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from torchvision.transforms import Resize, ToTensor, Normalize

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm

train_dir = '/opt/ml/input/data/train'
train_img_dir = os.path.join(train_dir, 'images')
train_img_sub_dirs = [os.path.join(train_img_dir, sub_dir) for sub_dir in os.listdir(train_img_dir) if os.path.isdir(os.path.join(train_img_dir, sub_dir))]

default_img_paths = np.array([[os.path.join(sub_dir, img) for img in os.listdir(sub_dir) if not img.startswith('.')]  for sub_dir in train_img_sub_dirs]).flatten()

default_transforms = transforms.Compose([
    transforms.ToTensor()
])

class TestDataset(Dataset):
    def __init__(self, img_paths, transform):
        self.img_paths = img_paths
        self.transform = transform

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])

        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.img_paths)

class TrainDataset(Dataset):
    def __init__(self, img_paths=default_img_paths, transforms=default_transforms):
        self.train_info = pd.read_csv(os.path.join(train_dir, 'train.csv'))
        self.train_info['label_weight'] = self._cal_label_weight(self.train_info['gender'], self.train_info['age'])
        
        self.img_paths = img_paths
        self.transforms = transforms

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        image = Image.open(img_path)
        tmp_dir, img_name = os.path.split(img_path)
        _, person = os.path.split(tmp_dir)
        
        if self.transforms:
            image = self.transforms(image)
        
        y = self._toY(person, img_name)
        return image, y

    def __len__(self):
        return len(self.img_paths)
    
    def _cal_label_weight(self, gender, age):
        weight = np.zeros(gender.shape)
        # gender
        weight += (gender == 'female')*3
        # age
        weight += ((age >= 30) & (age < 60))*1
        weight += (age >= 60)*2

        return weight
    
    def _toY(self, person, mask_label):
        label_weight = self.train_info.query(f"path == '{person}'")['label_weight'].values[0]
        mask_weight = 0
        if mask_label.startswith('incorrect'):
            mask_weight += 6
        elif mask_label.startswith('normal'):
            mask_weight += 12
        return label_weight + mask_weight
        
    
    def __repr__(self):
        idx = np.random.randint(len(self))
        X, y = self[idx]
        return f'[{self.__class__.__name__}]\n\t length : {len(self)} \n\t y : {y} \n\t X.shape : {X.shape} \n\t X : \n{X}'


train_set = TrainDataset()
train_img_iter_basic = DataLoader(train_set)
train_img_iter_batch = DataLoader(train_set,
                           batch_size=100
                           )
train_img_iter_numworker = DataLoader(train_set,
                           num_workers=3
                           )
train_img_iter_numworker_batch = DataLoader(train_set,
                            batch_size=20,
                            num_workers=2
                           )

