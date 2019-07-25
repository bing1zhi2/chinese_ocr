# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
from imp import reload
from PIL import Image
import numpy as np
import tensorflow as tf

from densenet import keys
from densenet import densenet
from config.use_model_config import DENSENET_MODEL_DIR

model_dir = DENSENET_MODEL_DIR
input_file = "demo.jpg"

reload(densenet)

characters = keys.alphabet[:]
characters = characters[1:] + u'卍'
nclass = len(characters)

"""
signature_def['predict_images']:
The given SavedModel SignatureDef contains the following input(s):
inputs['images'] tensor_info:
    dtype: DT_FLOAT
    shape: (-1, 32, -1, 1)
    name: the_input_2:0
The given SavedModel SignatureDef contains the following output(s):
outputs['prediction'] tensor_info:
    dtype: DT_FLOAT
    shape: (-1, -1, 5990)
    name: out_2/truediv:0
Method name is: tensorflow/serving/predict

"""
class DensenetOcr:
    def __init__(self):
        self.run_func = self.loadmodel()

    def loadmodel(self):
        with tf.Graph().as_default():
            sess = tf.Session()
            with sess.as_default():
                return self.run_lamada(sess)

    def run_lamada(self, sess):
        print("loading ocr model from " + model_dir+"...............")
        metagraph_def = tf.saved_model.loader.load(
            sess, [tf.saved_model.tag_constants.SERVING], model_dir)
        # print(metagraph_def)

        signature_def = metagraph_def.signature_def[
            "predict_images"]
        input_tensor = sess.graph.get_tensor_by_name(
            signature_def.inputs['images'].name)
        output_tensor = sess.graph.get_tensor_by_name(
            signature_def.outputs['prediction'].name)

        run_func = lambda images: output_tensor.eval(session=sess, feed_dict={input_tensor: images})

        return run_func

    def runimg(self, img):
        width, height = img.size[0], img.size[1]
        scale = height * 1.0 / 32
        width = int(width / scale)

        img = img.resize([width, 32], Image.ANTIALIAS)
        img = np.array(img).astype(np.float32) / 255.0 - 0.5

        X = img.reshape([1, 32, width, 1])

        y_pred = self.run_func(X)
        y_pred = y_pred[:, :, :]

        return y_pred

    def recognize(self, img):

        """
        the method which to use
        :param img:
        :return: chinese character that in the image
        """

        y_pred = self.runimg(img)
        y_pred = y_pred[:, :, :]

        # out = K.get_value(K.ctc_decode(y_pred, input_length=np.ones(y_pred.shape[0]) * y_pred.shape[1])[0][0])[:, :]
        # out = u''.join([characters[x] for x in out[0]])
        out = self.decode(y_pred)

        return out

    def decode(self, pred):
        char_list = []
        pred_text = pred.argmax(axis=2)[0]
        for i in range(len(pred_text)):
            if pred_text[i] != nclass - 1 and (
                    (not (i > 0 and pred_text[i] == pred_text[i - 1])) or (i > 1 and pred_text[i] == pred_text[i - 2])):
                # print(pred_text[i])
                char_list.append(characters[pred_text[i]])
        return u''.join(char_list)

    def img_2_id(self, img):
        """
        把图片文字运算成字符id,方便计算acc 使用
        :param img:
        :return:
        """
        y_pred = self.runimg(img)
        y_pred = y_pred[:, :, :]

        out = self.decode_to_id(y_pred)
        return out

    def decode_to_id(self, pred):
        """
        first decode to id ,then to char ,
         only to calculate acc ....you can directly use recognize
        :param pred:
        :return:
        """
        id_list = []
        pred_text = pred.argmax(axis=2)[0]
        for i in range(len(pred_text)):
            if pred_text[i] != nclass - 1 and (
                    (not (i > 0 and pred_text[i] == pred_text[i - 1])) or (i > 1 and pred_text[i] == pred_text[i - 2])):
                id_list.append(pred_text[i])
        return id_list


    def id_to_char(self, id_list):
        """
        id to char
        :param id_list:
        :return:
        """
        char_list = []
        for i in range(len(id_list)):
            char_list.append(characters[id_list[i]])
        return u''.join(char_list)

