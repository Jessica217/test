#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Introduce :
@File      : open_api_demo.py
@Time      : 2024/11/13 11:32
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
def init_openai_client(api_key="EMPTY", api_base= "http://localhost:8080/v1"):
    """
    初始化OpenAI客户端。

    参数：
        api_key (str): API密钥。
        api_base (str): API基础URL。

    返回：
        OpenAI: OpenAI客户端实例。
    """
    return OpenAI(api_key=api_key, base_url=api_base)


# 生成请求提示语（Prompt）
def get_prompt(vip_vuln_info):
    """
    根据漏洞信息生成提示语，用于提取CPE信息。

你是一位顶尖的网络安全专家，专注于分析和处理漏洞情报。请从给定的漏洞描述中提取符合CPE格式的所有网络安全CPE（通用平台枚举）。

    **提取要求：**
    1. 输出仅包含符合CPE格式的字符串：`cpe:/{{part}}:{{vendor}}:{{product}}:{{version}}:{{update}}:{{edition}}:{{language}}`。
    2. 如果原始CPE数据中缺少某些字段，保留其原样，不添加或修改任何信息。

    **输出格式：**
    - 将结果格式化为JSON对象，主键为`cpe`，值为包含每个CPE字符串的数组。
    - 不需要额外的解释或描述。

    ### 漏洞情报：

    参数：
        vip_vuln_info (str): 漏洞描述信息。

    返回：
        str: 格式化后的提示语。
    """
    prompt = f"""
You are a world-class cybersecurity expert specializing in the analysis and handling of vulnerability intelligence. From the provided vulnerability description, please extract all relevant network security CPEs (Common Platform Enumeration) that adhere to the CPE format. 

**Requirements for Extraction:** 
1. The output should consist only of strings in the exact CPE format: `cpe:/{{part}}:{{vendor}}:{{product}}:{{version}}:{{update}}:{{edition}}:{{language}}`.
2. If certain fields are missing in the original CPE data, retain the details as they are without adding or modifying any information.

**Output Specifications:** 
- Format the results as a JSON object with a single key, `cpe`, whose value is an array listing each extracted CPE string individually.
- No extra explanations or descriptions are required.

### Vulnerability Intelligence:"
    {vip_vuln_info}
    """
    return prompt


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
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": f"{prompt}"},
        ],
        temperature=0.7,
        top_p=0.8,
        max_tokens=512,
        extra_body={
            "repetition_penalty": 1.05,
            "guided_json": {
                "type": "object",
                "properties": {
                    "cpe": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                },
                "required": ["cpe"]
            }}
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
    openai_api_base = "http://10.0.81.159:8080/v1"

    # 初始化OpenAI客户端
    client = init_openai_client(openai_api_key, openai_api_base)

    # 漏洞描述信息
    vip_vuln_info = """VPL-JAIL-SYSTEM是jcrodriguez-dis个人开发者的一个库。为 VPL Moodle 插件提供了一个执行沙盒。
    VPL-JAIL-SYSTEM v4.0.2 版本及之前版本存在安全漏洞，该漏洞源于存在路径遍历问题。
    输出：漏洞影响的产品：VPL-JAIL-SYSTEM，漏洞影响的版本：<=v4.0.2
    """

    # 获取提示语
    prompt = get_prompt(vip_vuln_info)

    # 获取提取的CPE数据
    model_response = call_model_api(client, model="Qwen/Qwen2.5-14B-Instruct", prompt=prompt)
    cpe_data=json.loads(model_response)
    # 输出转换为JSON格式的CPE数据
    print("++++++++++++++++++++")
    for cpe_str in cpe_data["cpe"]:
        print(f"CPE字符串:\n {cpe_str}")
        print(f"CPE JSON:\n {json.dumps(cpe_to_json(cpe_str), indent=4)}")


# 执行主函数
if __name__ == "__main__":
    main()




