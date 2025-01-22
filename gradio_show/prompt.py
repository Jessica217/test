#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Introduce :
@File      : prompt.py
@Time      : 2024/11/19 15:03
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



def get_prompt_contain_detail(vip_vuln_info):
    """
    根据漏洞信息生成提示语，用于提取CPE信息。
    参数：
        vip_vuln_info (str): 漏洞描述信息。

    返回：
        str: 格式化后的提示语。
    """
    prompt = f"""


You are a world-class cybersecurity expert specializing in the analysis and processing of vulnerability intelligence. Please extract all valid **CPE (Common Platform Enumeration)** strings from the following vulnerability description. Extraction rules and output format are as follows:

---

### Extraction Rules

1. **Strict Adherence to the CPE Standard Format:**  
   CPE strings must strictly follow the format:  `cpe:/{{part}}:{{vendor}}:{{product}}:{{version}}:{{update}}:{{edition}}:{{language}}`， A total of 7 fields must be included. If any field is missing, it must be replaced with the placeholder `*`, ensuring all fields are complete.

2. **Strict Matching Requirements:**  
   - Only extract CPE strings that fully conform to the format above.  
   - Do not infer or supplement data from the original description. Only use the content provided.

3. **Field Source Explanation:**  
   Each extracted CPE string must include an explanation of its fields. For example:  
   - If fields such as `vendor` and `product` are derived from specific content in the description, explicitly indicate the extraction logic.  
   - If fields such as `version` are missing, use the placeholder `*` and specify the absence in the explanation.

---

### Output Requirements

1. **JSON Output Format:**  
   Return a JSON object containing two keys:  
   - `cpe`: An array listing all extracted CPE strings.  
   - `detail`: An array corresponding to the `cpe` array, providing explanations for each extracted CPE string.

2. **Consistency in Formatting:**  
   - Strings in the `cpe` array should be listed in the order they are extracted.  
   - Ensure all fields are complete with no missing or extra placeholders.  
   - Explanations in the `detail` array must be concise and clearly indicate the source of each field.

---

### Reference Examples

#### Input Example 1:

```json
{{
  "vuln_name": "Linux kernel security vulnerability",
  "vuln_desc": "Linux kernel is the open-source operating system kernel managed by the Linux Foundation. A security vulnerability exists in the Linux kernel, which attackers can exploit via the Frozen EBPF Map Race to escalate privileges.",
  "effect_scope": null
}}
```

#### Output Example 1:

```json
{{
  "cpe": [
    "cpe:/o:linux:linux_kernel:*:*:*:*"
  ],
  "detail": [
    {{
      "cpe": "cpe:/o:linux:linux_kernel:*:*:*:*",
      "explanation": "Extracted 'Linux kernel' as the product and 'Linux' as the vendor from the description. Fields for version, update, edition, and language are not provided, so placeholders '*' are used."
    }}
  ]
}}
```

#### Input Example 2:

```json
{{
  "vuln_name": "Industrial control software vulnerability",
  "vuln_desc": "Mitsubishi Electric's MC Works64 and Iconics' Genesis64 software both have privilege escalation vulnerabilities.",
  "effect_scope": "Industrial control software"
}}
```

#### Output Example 2:

```json
{{
  "cpe": [
    "cpe:/a:mitsubishi_electric:mc_works64:*:*:*:*",
    "cpe:/a:iconics:genesis64:*:*:*:*"
  ],
  "detail": [
    {{
      "cpe": "cpe:/a:mitsubishi_electric:mc_works64:*:*:*:*",
      "explanation": "Extracted 'Mitsubishi Electric' as the vendor and 'MC Works64' as the product from the description. Fields for version, update, edition, and language are not provided, so placeholders '*' are used."
    }},
   {{
      "cpe": "cpe:/a:iconics:genesis64:*:*:*:*",
      "explanation": "Extracted 'Iconics' as the vendor and 'Genesis64' as the product from the description. Fields for version, update, edition, and language are not provided, so placeholders '*' are used."
    }}
  ]
}}
```

---

### Extraction Task

Based on the rules above, please extract the CPE strings from the vulnerability description and provide an explanation for each result. Ensure the format is consistent, all fields are complete, and the output conforms to the specified JSON format.

### Vulnerability Description:

    {vip_vuln_info}
    """
    return prompt


def get_prompt_no_detail(vip_vuln_info):
    """
    根据漏洞信息生成提示语，用于提取CPE信息。
    参数：
        vip_vuln_info (str): 漏洞描述信息。

    返回：
        str: 格式化后的提示语。
    """
    prompt = f"""
Based on the rules above, please extract the CPE strings from the vulnerability description and provide an explanation for each result. Ensure the format is consistent, all fields are complete, and the output conforms to the specified JSON format.
You are a world-class cybersecurity expert specializing in the analysis and handling of vulnerability intelligence. Please extract all cybersecurity CPEs (Common Platform Enumerations) that conform to the **CPE standard** from the following vulnerability descriptions. The extraction rules and output format are as follows:

---

### Extraction Rules

1. **Strictly Follow the CPE Standard Format:**  
   CPE strings must strictly conform to the following format:  
   `cpe:/{{part}}:{{vendor}}:{{product}}:{{version}}:{{update}}:{{edition}}:{{language}}`, consisting of 7 fields. If any field is missing, it must be filled with the placeholder `*`, ensuring every field is complete.

2. **Strict Matching Requirements:**  
   - Only extract strings that fully comply with the above standard format.  
   - Do not infer or complete any content based on the original data; extraction should be based solely on the content provided in the vulnerability description.

---

### Output Requirements

1. **Output in JSON Format:**  
   Return a JSON object containing a single key `cpe`, whose value is an array listing all extracted CPE strings.

2. **Format Consistency:**  
   - Strings within the array should be listed in the order they were extracted.  
   - Ensure that all fields are complete, with no missing or extra placeholders.

---

### Reference Examples

#### Input Example 1:

```json
{{
  "vuln_name": "Linux kernel security vulnerability",
  "vuln_desc": "The Linux kernel is an open-source operating system kernel developed by the Linux Foundation in the United States. It has a security vulnerability that allows attackers to escalate privileges via the Frozen EBPF Map Race.",
  "effect_scope": null
}}
```

#### Output Example 1:

```json
{{
  "cpe": [
    "cpe:/o:linux:linux_kernel:*:*:*:*"
  ]
}}
```

#### Input Example 2:

```json
{{
  "vuln_name": "Industrial control software vulnerability",
  "vuln_desc": "Privilege escalation vulnerabilities exist in Mitsubishi Electric's MC Works64 and Iconics Genesis64 software.",
  "effect_scope": "Industrial control software"
}}
```

#### Output Example 2:

```json
{{
  "cpe": [
    "cpe:/a:mitsubishi_electric:mc_works64:*:*:*:*",
    "cpe:/a:iconics:genesis64:*:*:*:*"
  ]
}}
```

---

### Extraction Task

Based on the rules above, extract all CPE strings from the vulnerability description, ensure the format is consistent, and fields are complete. Return the result in JSON format as specified.

---



### Vulnerability Description:

    {vip_vuln_info}
    """
    return prompt


def generate_prompt(vip_vuln_info: str, detail=False) -> str:
    """根据漏洞信息生成提示语"""
    # get_prompt_no_detail 是您的自定义函数，这里直接调用
    if detail:
        prompt = get_prompt_contain_detail(vip_vuln_info)
    else:
        prompt = get_prompt_no_detail(vip_vuln_info)
    print(f"生成的提示语:\n{prompt}")
    return prompt

if __name__ == '__main__':

    vip_vuln_info =""" {
        'vuln_name': 'mingsoftmcms 使用硬编码凭据漏洞',
        'vuln_desc': 'MCMS v5.2.4 被发现存在硬编码的 shiro-key，允许攻击者利用该密钥执行任意代码。',
        'effect_scope': None
    }
    """
    prompt = generate_prompt(vip_vuln_info)