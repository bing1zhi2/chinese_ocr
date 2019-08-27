# -*- coding: utf-8 -*-

import os

"""
mode config 
"""


class InferenceConfig:
    # the mode path to use in demo
    DENSENET_MODEL_DIR = os.getcwd() + "/models/densenet_base_model/1"


class Dict1Config(InferenceConfig):
    print("config 5990 chars model path ")


class Dict2Config(InferenceConfig):
    print("config 7476 chars model")
    DENSENET_MODEL_DIR = os.getcwd() + "/models/densenet_base_model/2"

inference_config = {
    '5990_model': Dict1Config,
    '7476_model': Dict2Config
}