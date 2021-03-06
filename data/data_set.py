import os
import cv2
import pandas as pd
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from torchvision import transforms
from torchvision.transforms import Resize, ToTensor, Normalize


from config import trainDataImgPaths, trainDataDir

default_img_paths = trainDataImgPaths

default_transforms = transforms.Compose([
    Resize((512, 384), Image.BILINEAR),
    ToTensor(),
    Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),
])

class TestDataset(Dataset):
    def __init__(self, img_paths, transform=default_transforms):
        self.img_paths = img_paths
        self.transform = transform

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])

        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.img_paths)

class MergedTrainDataSet(Dataset):
    def __init__(self, img_paths=default_img_paths, transforms=default_transforms):
        self.train_info = pd.read_csv(os.path.join(trainDataDir, 'train_info_merged.csv'))
        
        self.img_paths = img_paths
        self.transforms = transforms

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        image = Image.open(img_path)
        
        if self.transforms:
            image = self.transforms(image)

        y = self.train_info[self.train_info.fullpath == img_path].agg_label.values[-1]
        return image, y

    def __len__(self):
        return len(self.img_paths)


class TrainDataset(Dataset):
    def __init__(self, img_paths=default_img_paths, transforms=default_transforms):
        self.train_info = pd.read_csv(os.path.join(trainDataDir, 'train.csv'))
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


class MaskTrainSet(TrainDataset):
    def __init__(self, img_paths=default_img_paths, transforms=default_transforms):
        super().__init__(img_paths=img_paths, transforms=transforms)
    
    def _toY(self, person, mask_label):
        mask_weight = 0
        if mask_label.startswith('incorrect'):
            mask_weight += 1
        elif mask_label.startswith('normal'):
            mask_weight += 2
        return mask_weight

class AgeTrainSet(TrainDataset):
    def __init__(self, img_paths=default_img_paths, transforms=default_transforms):
        super().__init__(img_paths=img_paths, transforms=transforms)
        age = self.train_info.age
        weight = ((age >= 30) & (age < 60))*1
        weight += (age >= 60)*2
        self.train_info['age'] = weight
    
    def _toY(self, person, mask_label):
        age = self.train_info.query(f"path == '{person}'")['age'].values[0]
        return age

class GenderTrainSet(TrainDataset):
    def __init__(self, img_paths=default_img_paths, transforms=default_transforms):
        super().__init__(img_paths=img_paths, transforms=transforms)
        self.train_info['gender'] = self.train_info.gender.map({'female': 1, 'male': 0})
    
    def _toY(self, person, mask_label):
        gender = self.train_info.query(f"path == '{person}'")['gender'].values[0]
        return gender
