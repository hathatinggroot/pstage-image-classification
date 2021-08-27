import os
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from torchvision.transforms import Resize, ToTensor, Normalize

import data.data_loader as D
import model.model as M
from tqdm.auto import tqdm


test_dir = '/opt/ml/input/data/eval'
model_dir = '/opt/ml/code/out/models'
out_dir = '/opt/ml/code/out/submission'

def submission(try_cnt: int, cat='', model_name=''):
    submission = pd.read_csv(os.path.join(test_dir, 'info.csv'))
    image_dir = os.path.join(test_dir, 'images')
    model_path = os.path.join(model_dir, cat, model_name)

    # Test Dataset 클래스 객체를 생성하고 DataLoader를 만듭니다.
    image_paths = [os.path.join(image_dir, img_id) for img_id in submission.ImageID]
    dataset = D.TestDataset(image_paths)

    loader = DataLoader(
        dataset,
        shuffle=False
    )

    # 모델을 정의합니다. (학습한 모델이 있다면 torch.load로 모델을 불러주세요!)
    device = torch.device('cuda')
    model = M.get_model(cat).to(device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)
    
    model.eval()

    # 모델이 테스트 데이터셋을 예측하고 결과를 저장합니다.
    all_predictions = []
    for images in tqdm(loader):
        with torch.no_grad():
            images = images.to(device)
            pred = model(images)
            pred = pred.argmax(dim=-1)
            all_predictions.extend(pred.cpu().numpy())
    submission['ans'] = all_predictions

    # 제출할 파일을 저장합니다.
    submission.to_csv(os.path.join(out_dir, f'submission_{str(try_cnt)}_{cat}.csv'), index=False)
    print('test inference is done!')

if __name__ == '__main__':
    # submission(5, 'mask', 'ResNet_2021-08-25_13:40:44.180349.pt')
    submission(5, 'age', 'ResNet_2021-08-26_03:48:17.512978+09:00.pt')
    submission(5, 'gender', 'ResNet_2021-08-26_05:20:58.302954+09:00.pt')