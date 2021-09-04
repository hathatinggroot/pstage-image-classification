
from torch.utils.data import DataLoader
import numpy as np
from sklearn.model_selection import StratifiedKFold

from data.data_set import *

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


mask_train_img_iter_numworker_batch = DataLoader(MaskTrainSet(),
                            batch_size=20,
                            num_workers=2
                           )

gender_train_img_iter_numworker_batch = DataLoader(GenderTrainSet(),
                            batch_size=20,
                            num_workers=2
                           )
age_train_img_iter_numworker_batch = DataLoader(AgeTrainSet(),
                            batch_size=20,
                            num_workers=2
                            )

def get_loader(data_set, batch_size: int = 20, num_workers:int = 2):
    return DataLoader(
        data_set,
        batch_size=batch_size,
        num_workers=num_workers
    )


def get_kfolded_loader_set(data_set, batch_size: int = 20, num_workers:int = 2, k: int = 5):
    n_samples = len(data_set)
    X = np.zeros(n_samples)
    y = data_set.train_info.agg_label.values
    
    skf = StratifiedKFold(n_splits=k)
    train_idx, valid_idx = next(iter(skf.split(X, y)))

    train_paths = data_set.img_paths[train_idx]
    valid_paths = data_set.img_paths[valid_idx]
    
    train_set = MergedTrainDataSet(img_paths=train_paths)
    valid_set = MergedTrainDataSet(img_paths=valid_paths)
    
    train_iter = DataLoader(
        train_set,
        batch_size=batch_size,
        num_workers=num_workers
    )

    valid_iter = DataLoader(
        valid_set,
        batch_size=batch_size,
        num_workers=num_workers
    )

    return train_iter, valid_iter

kfold_train_iter, kfold_valid_iter = get_kfolded_loader_set(MergedTrainDataSet())