# -*- coding:utf-8 -*-

def read_line(file_path, encoding='utf-8'):
    char_list=[]
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            char_list= f.readlines()
    except FileNotFoundError:
        print('找不到指定的文件!')
    except LookupError:
        print('指定了未知的编码!')
    except UnicodeDecodeError:
        print('读取文件时解码错误!')
    return char_list

def read_file(filename):
    res = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for i in lines:
            res.append(i.strip())
    dic = {}
    for i in res:
        p = i.split(' ')
        dic[p[0]] = p[1:]
    return dic
