#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Introduce :
@File      : vllm_niginx.py
@Time      : 2024/12/15 17:08
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
from openai import OpenAI


# 初始化 OpenAI API 客户端
def init_openai_client(api_key="EMPTY", api_base="http://localhost:8080/v1"):
    """
    初始化OpenAI客户端。

    参数：
        api_key (str): API密钥。
        api_base (str): API基础URL。

    返回：
        OpenAI: OpenAI客户端实例。
    """
    return OpenAI(api_key=api_key, base_url=api_base)


# 远程调用模型API
def call_model_api(client, model, prompt):
    """
    调用模型API以获取CPE数据。

    参数：
        client (OpenAI): OpenAI客户端实例。
        model (str): 使用的模型名称。
        prompt (str): 请求提示语。

    返回：
        dict: 提取的CPE数据。
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are Appsec bot, created by 软评中心. You are a helpful assistant."},
            {"role": "user", "content": f"{prompt}"},
        ],
        temperature=0.5,
        top_p=0.8,
        max_tokens=512,
        # extra_body={
        #     "repetition_penalty": 1.05,
        #     "guided_json": {
        #         "type": "object",
        #         "properties": {
        #             "cpe": {
        #                 "type": "array",
        #                 "items": {"type": "string"}
        #             },
        #
        #         },
        #         "required": ["cpe"]
        #     }
        # }
    )
    return response.choices[0].message.content


# 将CPE字符串转换为JSON格式
def cpe_to_json(cpe_str):
    """
    将CPE字符串转换为JSON格式。

    参数：
        cpe_str (str): CPE字符串。

    返回：
        dict: 转换后的JSON格式数据。
    """
    # 去掉前缀 "cpe:/"
    parts = cpe_str.replace("cpe:/", "").split(":")

    # CPE字段映射
    fields = ["part", "vendor", "product", "version", "update", "edition", "language"]

    # 将CPE字段与内容对应
    cpe_json = {fields[i]: parts[i] if i < len(parts) else "" for i in range(len(fields))}

    return cpe_json


# 主函数
def main():
    # 配置API密钥和API基础URL
    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:58082/v1"
    # 初始化OpenAI客户端
    client = init_openai_client(openai_api_key, openai_api_base)
    prompt = "你是谁"
    # 获取提取的CPE数据
    model_response = call_model_api(client, model="appsec_bot", prompt=prompt)
    print(f"model_response: \n{model_response}")




# 执行主函数
if __name__ == "__main__":
    main()
