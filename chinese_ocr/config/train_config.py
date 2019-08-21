# -*- coding:utf-8 -*-


class TrainConfig:
    # 训练图片的宽高
    IMG_H = 32
    IMG_W = 280
    BATCH_SIZE = 50
    MAX_LABEL_LENGTH = 10


    DATASET_SOURCE1="Synthtext"


    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")