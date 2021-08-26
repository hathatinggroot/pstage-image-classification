import os

# directory paths
rootDir = os.path.split(os.path.abspath(__file__))[0]
datasetDir = os.path.join(rootDir, "dataset")
hooksDir = os.path.join(rootDir, "hooks")
modelDir = os.path.join(rootDir, "model")
outDir = os.path.join(rootDir, "out")
outLogsDir = os.path.join(outDir, "logs")
outModelsDir = os.path.join(outDir, "models")
outSubmissionDir = os.path.join(outDir, "submission")







