
from torch.utils.data import DataLoader

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
                            num_workers=4
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