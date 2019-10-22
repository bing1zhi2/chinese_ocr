# -*- coding:utf-8 -*-
# bing1   

# 2019/8/21   

# 11:11   
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
"""
import sys
import os
import argparse


# dataset_url = "/media/chenhao/study/code/other/out"
# dict_file_name = "char_7476.txt"
# tmp_label_path = "/media/chenhao/study/code/other/out/default/tmp_labels.txt"

def main(dataset_url, dict_file_name, tmp_label_path):
    """
    把使用text_renderer 生成的label文件,转化为需要的格式

    :param dataset_url:
    :param dict_file_name:
    :param tmp_label_path:
    :return:
    """
    dict_file_path = os.path.join(dataset_url, dict_file_name)

    out_file_name = "my_dataset_label.txt"

    char_set = open(dict_file_path, 'r', encoding='utf-8').readlines()
    char_set = ''.join([ch.strip('\n') for ch in char_set][1:] + ['卍'])

    d = {' ': '0'}
    for i, c in enumerate(char_set):
        d[c] = str(i + 1)

    g = open(os.path.join(dataset_url, out_file_name), "w", encoding="utf-8")
    s = 0
    with open(tmp_label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # label = line.split(" ")
            (img_name, *line_text) = line.split(" ")
            words = " ".join(line_text)
            print(words)
            label_list = []
            if img_name =="00003580":
                print("dddd")
            for w in words:
                if w == ' ':
                    print("aaaa ", d[w])
                label_list.append(d[w])

            label_str = " ".join(label_list)
            print(label_str)
            g.write(img_name + ".jpg " + label_str + "\n")
            s += 1

    g.close()
    print("wirte ", s)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('dataset_url', type=str, help='text generate dataset url')
    parser.add_argument('dict_file_name', type=str, help='file name.')
    parser.add_argument('tmp_label_path', type=str, help='tmp_labels.txt path ')

    return parser.parse_args(argv)


if __name__ == '__main__':
    ### F:/code/other/text_renderer/output char_7473.txt F:/code/other/text_renderer/output/default/tmp_labels.txt
    a = sys.argv[1:]
    print("args:", a)
    args = parse_arguments(a)
    main(args.dataset_url, args.dict_file_name, args.tmp_label_path)
