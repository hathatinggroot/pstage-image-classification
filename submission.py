import os
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
from torchvision.transforms import Resize, ToTensor, Normalize
from tqdm.auto import tqdm
import datetime
from pytz import timezone

from data.data_loader import get_loader
from data.data_set import TestDataset
import model.model as M
from model.modeler import loadWithCheckpoint, getModel
from config import testDataDir, outModelsDir, outSubmissionDir



# test_dir = '/opt/ml/input/data/eval'
# model_dir = '/opt/ml/code/out/models'
# out_dir = '/opt/ml/code/out/submission'

def submission(expr_name: str, modelFrame: nn.Module, trained_model_name: str):
    submission = pd.read_csv(os.path.join(testDataDir, 'info.csv'))
    image_dir = os.path.join(testDataDir, 'images')

    # Test Dataset 클래스 객체를 생성하고 DataLoader를 만듭니다.
    image_paths = [os.path.join(image_dir, img_id) for img_id in submission.ImageID]
    dataset = TestDataset(image_paths)

    loader = get_loader(data_set=dataset, num_workers=1)

    # 모델을 정의합니다. (학습한 모델이 있다면 torch.load로 모델을 불러주세요!)
    device = torch.device('cuda')
    model = loadWithCheckpoint(modelFrame, trained_model_name).to(device)
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
    now = datetime.datetime.now(timezone('Asia/Seoul'))
    submission.to_csv(os.path.join(outSubmissionDir, f'submission_{expr_name}_{now}.csv'), index=False)
    print('test inference is done!')

if __name__ == '__main__':
    modelFrame = getModel('MyResnet50')(num_classes=18)()
    submission('Expr_Focal_Loss_Folded_Relay', modelFrame, 'Expr_Focal_Loss_Folded_Relay_2021-08-30_16:31:22.696706+09:00_.pt')
