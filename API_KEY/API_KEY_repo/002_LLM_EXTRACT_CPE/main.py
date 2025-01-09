from fastapi import FastAPI, HTTPException, Body
import uvicorn
import json
from openai import OpenAI
import configparser
from pydantic import BaseModel
from typing import Optional

# 初始化 FastAPI 应用
app = FastAPI()

# 读取配置文件
config = configparser.ConfigParser()
config.read('config.ini')

# 连接到 Qwen_Max 使用配置中的 API Key
client = OpenAI(
    api_key=config['OPENAI']['API_KEY'],
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def extract_cpe(content):
    # 找到 JSON 部分的开始和结束位置
    start = content.find('{')
    end = content.rfind('}') + 1
    # 如果找到了有效的 JSON 部分
    if start != -1 and end != -1:
        json_str = content[start:end]
        try:
            # 解析 JSON 字符串
            data = json.loads(json_str)

            # 从字典中提取 'cpe' 键对应的值
            cpe_list = data["cpe"]
            return cpe_list
        except json.JSONDecodeError:
            return "提供的内容中的 JSON 部分格式不正确。"
        except KeyError:
            return "JSON 数据中没有找到 'cpe' 键。"
    else:
        return "内容中没有找到有效的 JSON 格式。"

def get_prompt_no_detail(vip_vuln_info):
    """
    根据漏洞信息生成提示语，用于提取CPE信息。
    参数：
        vip_vuln_info (str): 漏洞描述信息。

    返回：
        str: 格式化后的提示语。
    """
    prompt = f"""
You are a world-class cybersecurity expert specializing in the analysis and handling of vulnerability intelligence. Please extract all cybersecurity CPEs (Common Platform Enumerations) that conform to the **CPE standard** from the following vulnerability descriptions. 
Only extract directly mentioned versions and details, and do not infer or assume any information not explicitly provided. Extraction rules and output format are as follows:

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

def generate_prompt(vip_vuln_info):
    """根据是否需要细节来选择不同的提示生成方式。"""
    return get_prompt_no_detail(vip_vuln_info)

#使用vulnerabilityInfo 方法，使得输入可以是json格式的。
class VulnerabilityInfo(BaseModel):
    vuln_name: str
    vuln_desc: str
    effect_scope: Optional[str] = None

@app.post("/extract_cpe")
async def extract_cpe(vip_vuln_info: VulnerabilityInfo, detail: bool = False):
    """从漏洞描述中提取CPE字符串的API端点。"""
    user_message = generate_prompt(vip_vuln_info.model_dump_json())
    print('====================================user_message==========================')
    print(user_message)
    content = await chat_with_qwen_max(client, 'qwen-max-latest', user_message)
    print('====================================content==========================')
    print(content)
    all_cpe = process_extracted_cpe(content)
    print('====================================cpe_list==========================')
    print(all_cpe)
    all_cpe_list = []
    for cpe_str in all_cpe:
        # 调用修复后的 cpe_to_json 函数
        cpe_json = cpe_to_json(cpe_str)
        all_cpe_list.append(cpe_json)
        if "error" in cpe_json:
            print(f"Skipping invalid CPE: {cpe_json['error']}")
            continue
        # 确保 JSON 格式正确

        print("Processing CPE:", cpe_json)
    print(all_cpe_list)

    return [all_cpe_list]

async def chat_with_qwen_max(client, model_name, user_message):
    """与OpenAI的Qwen Max模型交互，获取CPE提取结果。"""
    completion =  client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_message},
        ],
        temperature=0.5,
        top_p=0.8,
        max_tokens=512,
        extra_body={"repetition_penalty": 1.05}
    )
    return completion.choices[0].message.content

def process_extracted_cpe(content):
    # 找到 JSON 部分的开始和结束位置
    start = content.find('{')
    end = content.rfind('}') + 1

    # 如果找到了有效的 JSON 部分
    if start != -1 and end != -1:
        json_str = content[start:end]
        try:
            # 解析 JSON 字符串
            data = json.loads(json_str)

            # 从字典中提取 'cpe' 键对应的值
            cpe_list = data["cpe"]
            return cpe_list
        except json.JSONDecodeError:
            return "提供的内容中的 JSON 部分格式不正确。"
        except KeyError:
            return "JSON 数据中没有找到 'cpe' 键。"
    else:
        return "内容中没有找到有效的 JSON 格式。"

def cpe_to_json(cpe_str):
    # 定义 CPE 的字段名称
    fields = ["part", "vendor", "product", "version", "update", "edition", "language"]

    # 检查 CPE 格式
    if not cpe_str.startswith("cpe:/"):
        return {"error": f"Invalid CPE format: {cpe_str}"}

    # 解析 CPE 字符串，去除 "cpe:/" 并按 ":" 分割
    parts = cpe_str.replace("cpe:/", "").split(":")

    # 确保字段数量完整（不足时补充 "*"，多余时截断）
    if len(parts) > len(fields):
        return {"error": f"Invalid CPE format: {cpe_str} (too many fields)"}

    # 构建 JSON 对象，缺少的字段用 "*"
    cpe_json = {fields[i]: parts[i] if i < len(parts) else "*" for i in range(len(fields))}
    return cpe_json


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8010)
