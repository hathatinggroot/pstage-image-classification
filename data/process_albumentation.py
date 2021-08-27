import albumentations as A
import os
import cv2

from PIL import Image
from collections import Counter
import data_loader as D
from tqdm.auto import tqdm
from torchvision import transforms


def augment(img, img_path):
    img_dir, img_file = os.path.split(img_path)
    img_name, ext = img_file.split('.')
    for i in range(5):
        t_img = transform(image=img)["image"]
        new_path = os.path.join(img_dir, f'{img_name}_{i}.{ext}')
        cv2.imwrite(new_path, t_img)
        

if __name__ == '__main__':

    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Blur(p=0.4)
    ])

    img_paths = list(filter(lambda p: 'incorrect' in p or 'normal' in p , D.default_img_paths))
    
    for img_path in tqdm(img_paths):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        augment(img, img_path)
        
        
        