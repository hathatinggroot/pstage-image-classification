import os
import numpy as np

# train directory paths
rootDir = os.path.split(os.path.abspath(__file__))[0]
datasetDir = os.path.join(rootDir, "dataset")
hooksDir = os.path.join(rootDir, "hooks")
modelDir = os.path.join(rootDir, "model")
outDir = os.path.join(rootDir, "out")
outLogsDir = os.path.join(outDir, "logs")
outModelsDir = os.path.join(outDir, "models")
outSubmissionDir = os.path.join(outDir, "submission")


# data directory paths
dataDir = os.path.join(os.path.split(rootDir)[0], 'input', 'data')
## train data
trainDataDir = os.path.join(dataDir, 'train')
trainDataImgDir = os.path.join(trainDataDir, 'images_origin')
trainDataImgSubDirs = [os.path.join(trainDataImgDir, sub_dir) for sub_dir in os.listdir(trainDataImgDir) if os.path.isdir(os.path.join(trainDataImgDir, sub_dir))]
trainDataImgPaths = np.array([[os.path.join(sub_dir, img) for img in os.listdir(sub_dir) if not img.startswith('.')]  for sub_dir in trainDataImgSubDirs]).flatten()
## test data
testDataDir = os.path.join(dataDir, 'eval')
testDataImgDir = os.path.join(testDataDir, 'images')







