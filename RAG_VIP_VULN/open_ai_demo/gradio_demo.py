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
import random
import time
import gradio as gr

warnings.simplefilter("ignore")
# 显示所有列
pd.set_option('display.max_columns', 20)
# 显示所有行
pd.set_option('display.max_rows', 50)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 100)

# 定义一个函数，处理并返回排序后的前5行DataFrame
def get_data(user_input):
    print(f"user_input:\n{user_input}")
    print(f"Processing user input...")
    time.sleep(2)  # 模拟延迟
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
        data['score'] = random.randint(100 - index * 5, 105 - index * 5)  # 添加随机评分
        data['user_note'] = user_input  # 添加用户输入到表格
        data_dicts.append(data)

    # 转换为 DataFrame
    df = pd.DataFrame(data_dicts)

    # 根据 'score' 列降序排序，并添加排名列
    df = df.sort_values(by="score", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1  # 添加 rank 列，排名从 1 开始

    return df.head(5)


# 创建Gradio界面，增加输入框，并设置布局为上下
with gr.Blocks() as demo:
    with gr.Column():
        user_input = gr.Textbox(label="输入框：请输入vip漏洞情报内容", placeholder="输入您的文本")
        output_data = gr.Dataframe(label='Top 5 Processed Data (Sorted by Score)', interactive=False)
        submit_button = gr.Button("提交")

    # 点击提交按钮时调用 get_data 函数
    submit_button.click(fn=get_data, inputs=user_input, outputs=output_data)

# 启动界面
demo.launch(server_name='10.0.81.173', server_port=58082)
