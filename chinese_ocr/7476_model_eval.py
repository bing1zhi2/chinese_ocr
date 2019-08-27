# -- coding:utf-8 --
"""

计算单字准确率
"""
from keras import backend as K
import os
import shutil
from imp import reload
import tensorflow as tf
import cv2
from PIL import Image
import numpy as np

import matplotlib.pyplot as plt
from keras.layers import Input
from keras.models import Model
from keras.utils import multi_gpu_model

from chinese_ocr.train.train import random_uniform_num, get_session, get_model
from chinese_ocr.densenet_common import densenet
from chinese_ocr.densenet_common.densenet_model import data_generator
from chinese_ocr.train.synthtext_config import SynthtextConfig
from chinese_ocr.densenet_common.dataset_format import DataSetSynthtext
from predict_tf_tool import DensenetOcr

reload(densenet)
K.set_learning_phase(0)

dataset_path = "/media/chenhao/study/code/other/out"

class_id_file = "char_7476.txt"
train_label_name = "label_train.txt"
val_label_name = "label_val.txt"
test_label_name = "label_test.txt"
dataset_format = 0
sub_img_folder = "default"

data_test = DataSetSynthtext()
data_test.load_data(class_id_file, dataset_path, test_label_name, subset=sub_img_folder)
# label_list = data.load_instance(33)
# print(label_list)
data_test.prepare()

nclass = data_test.num_classes
char_set_line = data_test.char_set_line
print("class num:", nclass)
print("char_set_line:", char_set_line)
#
# input = Input(shape=(32, None, 1), name='the_input')
# # input = Input(tensor=resize_image)
# y_pred= densenet.dense_cnn(input, nclass)
# basemodel = Model(inputs=input, outputs=y_pred)
#
# modelPath = '/media/chenhao/study/code/work/github/chinese_ocr/chinese_ocr/models/weights_densenet-12-0.98.h5'
# # basemodel = multi_gpu_model(basemodel, gpus=8)
# print(modelPath)
# w = basemodel.get_weights()[2]
# print(w)
# print("loading weights..............")
# basemodel.load_weights(modelPath)
# w = basemodel.get_weights()[2]
# print(w)

config = SynthtextConfig()
config.display()
''''''
test_gen = data_generator(data_test, config=config)

print("test num_images", data_test.num_images)
# predict

ocr_predict_7476 = DensenetOcr("./train/char_7476.txt", model_name="7476_model")


def eva_batch():
    aaa = next(test_gen)
    # print(aaa)
    input_tuple = aaa[0]
    # print(input_tuple["the_input"].shape)
    batch_lines_img = input_tuple["the_input"]
    y_true = input_tuple["the_labels"]
    img_paths = input_tuple["img_paths"]

    batch_img_result = ocr_predict_7476.run_func(batch_lines_img)
    print(batch_img_result.shape)

    accs = []
    for i in range(config.BATCH_SIZE):
        one_line = batch_img_result[i, :, :]
        line_label = y_true[i, :]
        line_label = np.array(line_label).astype(int)
        one_line = np.expand_dims(one_line, axis=0)
        line_result = ocr_predict_7476.decode_to_id(one_line)

        print("y_pres      ", line_result)
        print("line_label  ", line_label)
        print("img_path    ", img_paths[i])
        print("id_to_char    ", ocr_predict_7476.id_to_char(line_result))

        """
        y_pres       [1, 360, 21, 5, 5, 175, 16, 36, 26, 258, 264]
        line_label   [  1 360  21   5 175  16  36  26 258 264]
        img_path     train/images/20459843_2752426851.jpg
                       的近20名大学生保安
        id_to_char     的近200名大学生保安
        有重复 [1, 360, 21, 5, 5, 175, 16, 36, 26, 258, 264]
        acc 0.4
        """
        # if len(line_result) > len(line_label):
        #     aaa = line_result.copy()
        #     # if remove_duplicates(aaa) == len(line_label):
        #     print("-------------->有重复", aaa)
        #     #     line_result = aaa[:len(line_label)]
        #     line_result = line_result[:len(line_label)]
        pre_arr = np.array(line_result)
        result = calculate_char_equal(line_label, pre_arr)
        totalRight = np.sum(result)
        acc = totalRight / len(result)
        print('acc', acc)
        accs.append(acc)
    print("batch mean acc", np.mean(acc))
    return accs


def calculate_char_equal(line_label, predict_arr):
    label_len = len(line_label)
    predict_len = predict_arr.shape[0]
    print("predict_len ", predict_len)
    # 长度不足的补字符
    if predict_len < label_len:
        print("predict shorter than true label")
        supplement = label_len - predict_len
        p_ = -1  # 补-1
        for i in range(supplement):
            predict_arr = np.append(predict_arr, p_)  # 直接向p_arr里添加p_
    # 过长的直接截取
    elif predict_len > label_len:
        print("predict longer than true label")
        predict_arr = predict_arr[:label_len]
    else:
        pass
    return np.equal(line_label, predict_arr)


accs = []
# epoch = data_test.num_images // config.BATCH_SIZE
epoch = 50
for i in range(epoch):
    print("---------------- eval epoch ----------", i)
    acc = eva_batch()
    accs += acc

print(accs)
prec = np.mean(accs)
print("total mean acc ", prec)
