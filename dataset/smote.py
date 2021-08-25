from imblearn.over_sampling import SMOTE
from PIL import Image
from collections import Counter
import data_loader as D

if __name__ == '__main__':
    for i , (inputs, labels) in enumerate(D.train_img_iter_batch):
        if i == 2:
            break
        size = inputs.shape[0]

        print(inputs.reshape(size, -1).shape)
        c = Counter(labels.numpy())
        print(f'before: {c}')
        oversample = SMOTE(k_neighbors=1)
        inputs, labels = oversample.fit_resample(inputs.reshape(size, -1), labels)
        c_ = Counter(labels)
        print(f'after: {c_}')