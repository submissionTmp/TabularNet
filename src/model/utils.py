# -*- coding: utf-8 -*-
import os
from os import path as osp
import pickle
from jsmin import jsmin
import json


def load_pickle(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)


def save_pickle(data, file_path):
    with open(file_path, "wb") as f:
        data = pickle.dump(data, f)


def load_json(file_path):
    with open(file_path) as js_file:
        s = jsmin(js_file.read())
    return json.loads(s)


def dic_add(dic, key, add_num=1):
    if key not in dic:
        dic[key] = add_num
    else:
        dic[key] += add_num


def append_to_file(file_path, s):
    with open(file_path, "a") as f:
        f.write(s)
