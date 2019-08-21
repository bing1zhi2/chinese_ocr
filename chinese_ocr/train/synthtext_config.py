
from chinese_ocr.config.train_config import TrainConfig

class SynthtextConfig(TrainConfig):
    DATASET_SOURCE1 = "Synthtext"
    BATCH_SIZE=20