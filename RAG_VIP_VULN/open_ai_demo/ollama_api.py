#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Introduce :
@File      : ollama_t4.py
@Time      : 2024/8/18 21:53
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
from ollama import Client
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
# 配置 Ollama 的 LLM 环境变量
url = 'http://10.0.81.173:11434'

client = Client(host=url)
vip_vuln_info="""
vuln_name: lexmark多个产品 命令注入漏洞'
vuln_desc: Embedded web server command injection vulnerability in Lexmark devices through 2021-12-07.
    """

    # 获取提示语
prompt = get_prompt(vip_vuln_info)
print(f"prompt:\n{prompt}")
model="qwen2.5:72b"
response = client.chat(model=model,
                       format= "json",

                       messages=[
                           {"role": "system", "content": "You are a helpful assistant."},
                           {
                               'role': 'user',
                               'content': f'{prompt}',
                           },
                       ])
print(f"response: {response}")
print(f"response type : {type(response)}")
from pprint import pprint

pprint(response)

content=response['message']['content']
print(f"content:\n{content}")

cpe_data = json.loads(content)

# 输出转换为JSON格式的CPE数据
from cpe2json import cpe_to_json

print("++++++++++++++++++++")
for cpe_str in cpe_data["cpe"]:
    print(f"CPE字符串:\n {cpe_str}")
    print(f"CPE JSON:\n {cpe_to_json(cpe_str)}")
