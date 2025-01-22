#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Introduce :
@File      : gradio_demo.py
@Time      : 2024/11/18 9:47
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

import gradio as gr
import pandas as pd
import json
import random
import time


# 定义一个函数，处理并返回DataFrame
def get_data(user_input):
    """
    user_input:漏洞情报内容
    """
    print(f"user_input:\n{user_input}")
    print(f"process user input....... ")
    time.sleep(2)
    # 原始数据
    raw_data = [
        "{\n    \"part\": \"a\",\n    \"vendor\": \"citrix\",\n    \"product\": \"application_delivery_controller\",\n    \"version\": \"12.0-fips\",\n    \"update\": \"\",\n    \"edition\": \"\",\n    \"language\": \"\"\n}",
        "{\n    \"part\": \"a\",\n    \"vendor\": \"citrix\",\n    \"product\": \"application_delivery_controller\",\n    \"version\": \"12.1-55.247\",\n    \"update\": \"\",\n    \"edition\": \"\",\n    \"language\": \"\"\n}",
        "{\n    \"part\": \"h\",\n    \"vendor\": \"citrix\",\n    \"product\": \"netscaler_application_delivery_controller\",\n    \"version\": \"-\",\n    \"update\": \"\",\n    \"edition\": \"\",\n    \"language\": \"\"\n}",
        "{\n    \"part\": \"a\",\n    \"vendor\": \"citrix\",\n    \"product\": \"netscaler_application_delivery_controller\",\n    \"version\": \"11.0\",\n    \"update\": \"\",\n    \"edition\": \"\",\n    \"language\": \"\"\n}",
        "{\n    \"part\": \"a\",\n    \"vendor\": \"citrix\",\n    \"product\": \"netscaler_application_delivery_controller\",\n    \"version\": \"11.1\",\n    \"update\": \"\",\n    \"edition\": \"\",\n    \"language\": \"\"\n}"
    ]

    # 将 JSON 字符串解析为字典列表
    data_dicts = []
    for index, item in enumerate(raw_data, 1):
        data = json.loads(item)
        data['score'] = random.randint(100 - index * 5, 105 - index * 5)
        # 将用户输入添加到表格数据中
        data_dicts.append(data)

    # 转换为 DataFrame
    df = pd.DataFrame(data_dicts)

    return df


# 创建Gradio界面，增加输入框，并设置布局为上下
with gr.Blocks() as demo:
    with gr.Column():
        user_input = gr.Textbox(label="输入框：请输入vip漏洞情报内容")
        output_data = gr.Dataframe(label='Processed Data 并返回CPE', interactive=False)
        submit_button = gr.Button("提交")

    submit_button.click(fn=get_data, inputs=user_input, outputs=output_data)

# 启动界面
demo.launch(server_name='0.0.0.0', server_port=58081)
