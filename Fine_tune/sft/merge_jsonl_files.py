#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Introduce :
@File      : merge_jsonl_files.py
@Time      : 2024/6/10 7:06
@Author    : wei.xu
@Tel       : 
@Email     : wei.xu@tophant.com
@pip       : pip install 
"""
import sys
import os
import time
import datetime
import numpy as np
import pandas as pd
import warnings
import json
from pathlib import Path

warnings.simplefilter("ignore")
# 显示所有列
pd.set_option('display.max_columns', 20)
# 显示所有行
pd.set_option('display.max_rows', 50)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 100)
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
parentPath = os.path.split(rootPath)[0]
grandFatherPath = os.path.split(parentPath)[0]
sys.path.append(curPath)
sys.path.append(rootPath)
sys.path.append(parentPath)
sys.path.append(grandFatherPath)

import json

def merge_jsonl_files(input_files, output_file):
    with open(output_file, 'wt+', encoding='utf-8') as outfile:
        for file in input_files:
            with open(f"{file}", 'rt+', encoding='utf-8',errors='ignore') as infile:
                for line in infile:
                    json_obj = json.loads(line.strip())
                    json.dump(json_obj, outfile, ensure_ascii=False)
                    outfile.write('\n')

# 定义要合并的JSONL文件列表
input_files = [
                '/data/xw/Qwen2/examples/sft/dataset/ruozhiba_qa2449_gpt4o_converted_data.jsonl',
                '/data/xw/Qwen2/examples/sft/dataset/identity_tophant_converted_data.jsonl',
               '/data/xw/Qwen2/examples/sft/dataset/wooyun_key_info.jsonl',
               '/data/xw/Qwen2/examples/sft/dataset/wooyun_labels_conversation.jsonl'
               ]
output_file = 'merged_output_202406110.jsonl'

# 调用函数合并文件
merge_jsonl_files(input_files, output_file)
