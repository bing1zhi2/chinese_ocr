# -*- coding:utf-8 -*-

import os
import cv2
import json
import time
import uuid
import base64
import web
from PIL import Image

web.config.debug = True
import model
from apphelper.image import union_rbox, adjust_box_to_origin
from global_obj import ocr_predict


detectAngle = True
detectAngle = False
path = "test_images/img.jpeg"
path = "test_images/line.jpg"

img = cv2.imread(path)  ##GBR
H, W = img.shape[:2]
timeTake = time.time()

# 多行识别  multi line
# _, result, angle = model.model(img,
#                                detectAngle=detectAngle,  ##是否进行文字方向检测，通过web传参控制
#                                config=dict(MAX_HORIZONTAL_GAP=50,  ##字符之间的最大间隔，用于文本行的合并
#                                            MIN_V_OVERLAPS=0.6,
#                                            MIN_SIZE_SIM=0.6,
#                                            TEXT_PROPOSALS_MIN_SCORE=0.1,
#                                            TEXT_PROPOSALS_NMS_THRESH=0.3,
#                                            TEXT_LINE_NMS_THRESH=0.7,  ##文本行之间测iou值
#                                            ),
#                                leftAdjust=True,  ##对检测的文本行进行向左延伸
#                                rightAdjust=True,  ##对检测的文本行进行向右延伸
#                                alph=0.01,  ##对检测的文本行进行向右、左延伸的倍数
#                                )
#
# print(result, angle)



# img = cv2.imread(path) ##GBR
## 单行识别 one line
partImg = Image.fromarray(img)
text2 = ocr_predict.recognize(partImg.convert('L'))
print(text2)