# -*- coding:utf-8 -*-
"""
把dict 文件转化成 一行字典
"""
import os

from chinese_ocr.utils.file_read import read_line
from densenet_common import keys


class DictKey:
    def __init__(self, dict_file_path=None):
        self.keys = self.init_key(dict_file_path)

    def init_key(self, dict_path):
        if dict_path:
            char_set = read_line(dict_path, encoding='utf-8')
            assert len(char_set) > 0
            char_set_temp = []
            for i, ch in enumerate(char_set):
                one_word = ch.strip('\n')
                char_set_temp.append(one_word)

            char_set_line = ''.join(char_set_temp[1:] + ['卍'])
            return char_set_line
        else:
            characters = keys.alphabet[:]
            characters = characters[1:] + u'卍'
            return characters

# a = DictKey("/media/chenhao/study/code/work/github/chinese_ocr/chinese_ocr/train/char_std_5990.txt")
# print(a.keys)
#
# b = DictKey()
# print(b.keys)
#
# c = DictKey("/media/chenhao/study/code/work/github/chinese_ocr/chinese_ocr/train/char_7476.txt")
# c = DictKey("chinese_ocr/train/char_7476.txt")
# print(os.getcwd())
# print(c.keys)
