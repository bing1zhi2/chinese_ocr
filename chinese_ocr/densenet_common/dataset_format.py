# -*-coding:utf-8 -*-
"""
此模块用于预定义 若干 数据集格式及对应的处理方式

继承 DataSet 类,按格式解析加载数据
"""
import os

from chinese_ocr.densenet_common.model_utils import Dataset
from chinese_ocr.utils.file_read import read_file

# from config.train_config import DATASET_SOURCE1
from chinese_ocr.utils.file_read import read_line

source = "Synthtext"


class DataSetSynthtext(Dataset):
    """
    格式 0 ,
    一个定义所有字符集的文件
    char_std_5990.txt
    数据集路径    一个标签文件,默认文件名
    20456062_1743140291.jpg 153 432 950 150 65 899 115 7 97 49
    """

    def load_data(self, class_file_name, data_set_dir, label_file_name, subset="images"):
        # 'char_std_5990.txt'  class_id_file, "./", "data_train.txt"
        class_file_path = os.path.join(data_set_dir, class_file_name)
        char_set = read_line(class_file_path, encoding='utf-8')
        char_set_temp = []
        for i, ch in enumerate(char_set):
            one_word = ch.strip('\n')
            self.add_class(source, i, one_word)
            char_set_temp.append(one_word)

        char_set_line = ''.join(char_set_temp[1:] + ['卍'])

        # char_set = ''.join([ch.strip('\n') for ch in char_set][1:] + ['卍'])
        # print(char_set)
        print(char_set_line)
        nclass = len(char_set_line)
        print(nclass)

        #

        annotions_file = os.path.join(data_set_dir, label_file_name)


        images_path = os.path.join(data_set_dir, subset)

        image_label = read_file(annotions_file)

        for key in image_label:
            self.add_image(source,
                           key,
                           os.path.join(images_path, key),
                           nclass=nclass,
                           label_list=image_label[key])

    def load_instance(self, image_id):

        # If not a DATASET_SOURCE1 dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != source:
            return super(self.__class__, self).load_instance(image_id)

        info = self.image_info[image_id]
        print(info)
        return info['label_list']

# data = DataSetSynthtext()
# data.load_data("../train/char_std_5990.txt", "../train", "data_train.txt")
# print("done")
# label_list = data.load_instance(33)
# print(label_list)
# data.prepare()
#
# print("Train Image Count: {}".format(len(data.image_ids)), data.num_images)
# print("Train Class Count: {}".format(data.num_classes))
# # for i, info in enumerate(data.class_info):
# #     print("{:3}. {:50}".format(i, info['name']))
