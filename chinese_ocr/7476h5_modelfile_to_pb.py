# -*- coding:utf-8 -*-
# bing1   

# 2019/8/26   

# 13:52   
"""
MIT License

Copyright (c) 2019 Hyman Chen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.



把用7476字典数据集训练的模型使用saved_model存成pb文件
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
from chinese_ocr.densenet_common.dataset_format import DataSetSynthtext


def run_func(model,img):
    return model.predict(img)

def deal_img(img):
    width, height = img.size[0], img.size[1]
    print(width, height)
    scale = height * 1.0 / 32
    width = int(width / scale)
    print(width, 32)

    img = img.resize([width, 32], Image.ANTIALIAS)

    plt.imshow(img)
    plt.show()

    img = np.array(img).astype(np.float32) / 255.0 - 0.5

    X = img.reshape([1, 32, width, 1])

    return X

    # y_pred = self.run_func(X)
    # y_pred = y_pred[:, :, :]
    #
    # return y_pred

def decode(pred, characters):
    char_list = []
    pred_text = pred.argmax(axis=2)[0]
    for i in range(len(pred_text)):
        if pred_text[i] != nclass - 1 and (
                (not (i > 0 and pred_text[i] == pred_text[i - 1])) or (i > 1 and pred_text[i] == pred_text[i - 2])):
            print(pred_text[i])
            char_list.append(characters[pred_text[i]])
    return u''.join(char_list)

def predict(img):
    """
    the method which to use
    :param img:
    :return: chinese character that in the image
    """

    y_pred = deal_img(img)

    y_pred = y_pred[:, :, :]

    # out = K.get_value(K.ctc_decode(y_pred, input_length=np.ones(y_pred.shape[0]) * y_pred.shape[1])[0][0])[:, :]
    # out = u''.join([characters[x] for x in out[0]])

    # out = decode(y_pred)
    # return out



reload(densenet)
K.set_learning_phase(0)


dataset_path="/media/chenhao/study/code/other/out"

class_id_file="char_7476.txt"
train_label_name="label_train.txt"
val_label_name="label_val.txt"
dataset_format= 0
sub_img_folder="default"

data = DataSetSynthtext()
data.load_data(class_id_file, dataset_path, train_label_name, subset=sub_img_folder)
# label_list = data.load_instance(33)
# print(label_list)
data.prepare()


nclass = data.num_classes
char_set_line = data.char_set_line
print("class num:",nclass)
print("char_set_line:", char_set_line)

input = Input(shape=(32, None, 1), name='the_input')
# input = Input(tensor=resize_image)
y_pred= densenet.dense_cnn(input, nclass)
basemodel = Model(inputs=input, outputs=y_pred)

modelPath = '/media/chenhao/study/code/work/github/chinese_ocr/chinese_ocr/models/weights_densenet-12-0.98.h5'
# basemodel = multi_gpu_model(basemodel, gpus=8)
print(modelPath)
w = basemodel.get_weights()[2]
print(w)

basemodel.load_weights(modelPath)
w = basemodel.get_weights()[2]
print(w)


''''''

# predict 
path = "test_images/line.jpg"

path = "test_images/line.jpg"
path = "test_images/20456062_1743140291.jpg"
path = "test_images/aaa.png"
path = "test_images/00582036.jpg"
# path = "test_images/00000028.jpg"
# path = "test_images/00349227.jpg"

img = cv2.imread(path)  ##GBR
# 单行识别 one line
partImg = Image.fromarray(img)
# text2 = recognize(partImg.convert('L'))

# text2 = recognize()

X = deal_img(partImg.convert('L'))
y_pred = basemodel.predict(X)

y_pred = y_pred[:, :, :]

result = decode(y_pred, char_set_line)

print(result)





# output

export_path = './model/2/'
if os.path.exists(export_path):
    shutil.rmtree(export_path)
version = 2
path='model'
K.set_learning_phase(0)
if not os.path.exists(path):
    os.mkdir(path)
export_path = os.path.join(tf.compat.as_bytes(path),tf.compat.as_bytes(str(version)))
builder = tf.saved_model.builder.SavedModelBuilder(export_path)
print('input info:', basemodel.input_names, '---', basemodel.input)
print('output info:', basemodel.output_names, '---', basemodel.output)
input_tensor = basemodel.input
output_tensor=basemodel.output

# tensor_info_input = tf.saved_model.utils.build_tensor_info(raw_image)

model_input = tf.saved_model.utils.build_tensor_info(input_tensor)
model_output = tf.saved_model.utils.build_tensor_info(output_tensor)

# im_info_output = tf.saved_model.utils.build_tensor_info(im_info)
# print('im_info_input tensor shape', im_info.shape)


prediction_signature = (
tf.saved_model.signature_def_utils.build_signature_def(
inputs={'images': model_input},
outputs={'prediction': model_output},
method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))


with K.get_session() as sess:
    builder.add_meta_graph_and_variables(
    sess=sess, tags=[tf.saved_model.tag_constants.SERVING],
    signature_def_map={'predict_images':prediction_signature,}
    )
    builder.save()
    
