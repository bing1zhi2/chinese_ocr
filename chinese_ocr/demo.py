# -*- coding:utf-8 -*-

import sys
import os
import time

from PIL import Image
import cv2

import model
from utils.image import union_rbox, adjust_box_to_origin
from utils.image import get_boxes, letterbox_image
from global_obj import ocr_predict
from config.config import opencvFlag, GPU, IMGSIZE, ocrFlag


def test_img1():
    detectAngle = True
    path = "test_images/shi.png"

    img = cv2.imread(path)  ##GBR
    H, W = img.shape[:2]
    timeTake = time.time()

    # 多行识别  multi line
    _, result, angle = model.model(img,
                                   detectAngle=detectAngle,  ##是否进行文字方向检测，通过web传参控制
                                   config=dict(MAX_HORIZONTAL_GAP=50,  ##字符之间的最大间隔，用于文本行的合并
                                               MIN_V_OVERLAPS=0.6,
                                               MIN_SIZE_SIM=0.6,
                                               TEXT_PROPOSALS_MIN_SCORE=0.1,
                                               TEXT_PROPOSALS_NMS_THRESH=0.3,
                                               TEXT_LINE_NMS_THRESH=0.7,  ##文本行之间测iou值
                                               ),
                                   leftAdjust=True,  ##对检测的文本行进行向左延伸
                                   rightAdjust=True,  ##对检测的文本行进行向右延伸
                                   alph=0.2,  ##对检测的文本行进行向右、左延伸的倍数
                                   )

    print(result, angle)
    result = union_rbox(result, 0.2)
    print(result, angle)

    res = [{'text': x['text'],
            'name': str(i),
            'box': {'cx': x['cx'],
                    'cy': x['cy'],
                    'w': x['w'],
                    'h': x['h'],
                    'angle': x['degree']

                    }
            } for i, x in enumerate(result)]
    res = adjust_box_to_origin(img, angle, res)  ##修正box
    print("res \n")
    print(res)

    print("result str: \n", )
    for x in result:
        print(x['text'])

    print("done")


def test_img2():
    """
    single line img test
    :return:
    """
    # detectAngle = False
    path = "test_images/line.jpg"

    img = cv2.imread(path)  ##GBR
    # 单行识别 one line
    partImg = Image.fromarray(img)
    text2 = ocr_predict.recognize(partImg.convert('L'))
    print(text2)
    # 客店遒劲摊婕有力


def test_img3():
    """
    single line need resize img, set alph to 0.2 to adjust anchor
    :return:
    """
    detectAngle = False
    path = "test_images/line.jpg"

    img = cv2.imread(path)  ##GBR
    img2, f = letterbox_image(Image.fromarray(img), IMGSIZE)

    _, result, angle = model.model(img2,
                                   detectAngle=detectAngle,  ##是否进行文字方向检测，通过web传参控制
                                   config=dict(MAX_HORIZONTAL_GAP=50,  ##字符之间的最大间隔，用于文本行的合并
                                               MIN_V_OVERLAPS=0.6,
                                               MIN_SIZE_SIM=0.6,
                                               TEXT_PROPOSALS_MIN_SCORE=0.1,
                                               TEXT_PROPOSALS_NMS_THRESH=0.3,
                                               TEXT_LINE_NMS_THRESH=0.7,  ##文本行之间测iou值
                                               ),
                                   leftAdjust=True,  ##对检测的文本行进行向左延伸
                                   rightAdjust=True,  ##对检测的文本行进行向右延伸
                                   alph=0.2,  ##对检测的文本行进行向右、左延伸的倍数
                                   )
    #
    print(result, angle)
    # [{'cx': 280.5, 'cy': 26.5, 'text': '客店遒劲摊婕有力', 'w': 606.0, 'h': 50.0, 'degree': 0.10314419109384157}] 0



def test_img4():
    detectAngle = True
    path = "test_images/fp.jpg"

    img = cv2.imread(path)  ##GBR
    H, W = img.shape[:2]
    timeTake = time.time()

    # 多行识别  multi line
    _, result, angle = model.model(img,
                                   detectAngle=detectAngle,  ##是否进行文字方向检测，通过web传参控制
                                   config=dict(MAX_HORIZONTAL_GAP=50,  ##字符之间的最大间隔，用于文本行的合并
                                               MIN_V_OVERLAPS=0.6,
                                               MIN_SIZE_SIM=0.6,
                                               TEXT_PROPOSALS_MIN_SCORE=0.1,
                                               TEXT_PROPOSALS_NMS_THRESH=0.3,
                                               TEXT_LINE_NMS_THRESH=0.7,  ##文本行之间测iou值
                                               ),
                                   leftAdjust=True,  ##对检测的文本行进行向左延伸
                                   rightAdjust=True,  ##对检测的文本行进行向右延伸
                                   alph=0.2,  ##对检测的文本行进行向右、左延伸的倍数
                                   )

    print(result, angle)
    result = union_rbox(result, 0.2)
    print(result, angle)

    res = [{'text': x['text'],
            'name': str(i),
            'box': {'cx': x['cx'],
                    'cy': x['cy'],
                    'w': x['w'],
                    'h': x['h'],
                    'angle': x['degree']

                    }
            } for i, x in enumerate(result)]
    res = adjust_box_to_origin(img, angle, res)  ##修正box
    print("res \n")
    print(res)

    print("result str: \n", )
    for x in result:
        print(x['text'])

    print("done")


test_img1()
