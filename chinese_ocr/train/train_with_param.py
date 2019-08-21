# -*- coding:utf-8 -*-

# from importlib import reload
import tensorflow as tf
from keras import losses
from keras import backend as K
from keras.utils import plot_model
from keras.preprocessing import image
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dense, Flatten
from keras.layers.core import Reshape, Masking, Lambda, Permute
from keras.layers.recurrent import GRU, LSTM
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, Adam
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, TensorBoard

from chinese_ocr.densenet_common.dataset_format import DataSetSynthtext
from chinese_ocr.train.synthtext_config import SynthtextConfig
from chinese_ocr.config.train_config import TrainConfig
from chinese_ocr.densenet_common.densenet_model import DensenetModel


#
# IMG_H = 32
#     IMG_W = 280
#     BATCH_SIZE = 50
#     MAX_LABEL_LENGTH = 10
#
def train_with_param(dataset_path, epochs, batch_size, dataset_format=0,
                     class_id_file="char_std_5990.txt", train_label_name="data_train.txt",
                     val_label_name="data_test.txt",sub_img_folder="images", max_label_length=10,
                     img_height=32, img_weight=280):
    config = TrainConfig()
    if dataset_format == 0:
        config = SynthtextConfig()
        config.BATCH_SIZE = batch_size
        config.MAX_LABEL_LENGTH = max_label_length
        config.IMG_H = img_height
        config.IMG_W = img_weight

    config.display()

    # class_id_file = "char_std_5990.txt"
    # train_label_name = "data_train.txt"
    # val_label_name = "data_test.txt"
    # sub_img_folder = "images"

    data = DataSetSynthtext()
    data.load_data(class_id_file, dataset_path, train_label_name, subset=sub_img_folder)
    # label_list = data.load_instance(33)
    # print(label_list)
    data.prepare()

    data_val = DataSetSynthtext()
    data_val.load_data(class_id_file, dataset_path, val_label_name, subset=sub_img_folder)
    data_val.prepare()

    print("Train Image Count: {}".format(len(data.image_ids)), data.num_images)
    print("Train Class Count: {}".format(data.num_classes))

    print("Val Image Count: {}".format(len(data_val.image_ids)), data_val.num_images)
    print("Val Class Count: {}".format(data_val.num_classes))

    # for i, info in enumerate(data.class_info):
    #     print("{:3}. {:50}".format(i, info['name']))

    model = DensenetModel(config=config, model_dir="./models")
    model.train(train_dataset=data, val_dataset=data_val, epochs=epochs)
