import os
import json
import re
from ollama import Client
from pprint import pprint

# 配置 Ollama 的 LLM 环境变量
url = 'http://10.0.81.173:11434'
client = Client(host=url)

# 指定路径
input_dir = "/root/data/wjy/vip_vul_pro/RAG_VIP_VULN/llm/output_cpe"
output_files = [f for f in os.listdir(input_dir) if f.startswith("output") and f.endswith(".txt")]

# 提取信息的正则表达式
vuln_pattern = r"### Vulnerability Intelligence:\n(.*?)\n\n"
ai_output_pattern = r"### AI 输出:\n```json\n(.*?)\n```"


def extract_vulnerability_info(file_content):
    vulnerability_intelligence = re.search(vuln_pattern, file_content, re.DOTALL)
    ai_output = re.search(ai_output_pattern, file_content, re.DOTALL)

    if vulnerability_intelligence and ai_output:
        return vulnerability_intelligence.group(1), json.loads(ai_output.group(1))
    else:
        return None, None


# 生成提示语函数
def get_prompt(vip_vuln_info, ai_output):
    """
    根据漏洞信息生成提示语，用于验证CPE输出的准确性。
    """
    prompt = f"""你是一位顶尖的网络安全专家，专注于分析和处理漏洞情报。根据提供的漏洞描述和大模型的输出，判断模型的输出是否正确，并根据以下要求输出结果。

### 任务要求：
1. **判断输出准确性**：根据原始漏洞描述，核对大模型输出的 CPE 是否准确。
2. **生成正确答案**：如果模型的 CPE 输出不正确，请基于漏洞描述，生成符合 CPE 标准的正确 CPE。

### 判断依据：
- 模型输出的 CPE 必须符合以下格式：`cpe:/{{part}}:{{vendor}}:{{product}}:{{version}}:{{update}}:{{edition}}:{{language}}`。
- 从漏洞描述中提取关键信息（如产品名称、版本、影响范围等），验证模型输出是否与描述匹配。
- 如果漏洞描述缺乏足够的信息，保持字段为空或省略，不得添加或虚构信息。

### 输出格式：
请以 JSON 对象格式输出，包含以下三个字段：
- `flag`: 布尔值。`true` 表示模型输出正确，`false` 表示不正确。
- `detail`: 详细解释判断依据，包括输出是否准确的原因、是否符合描述，以及正确答案的推导过程。
- `cpe`: 一个数组，包含从漏洞描述中提取或生成的正确 CPE 字符串。

### 输入信息：
#### 原始漏洞情报：
{vip_vuln_info}

#### 大模型输出：
{json.dumps(ai_output)}

请根据以上信息进行详细分析并输出结果。"""
    return prompt


# 调用 qwen-72b 模型的函数
def query_qwen_72b(vulnerability_intelligence, ai_output):
    # 构造 prompt
    prompt = get_prompt(vulnerability_intelligence, ai_output)
    print(f"prompt:\n{prompt}")

    # 调用 Ollama 的 qwen-72b 模型
    response = client.chat(
        model="qwen2.5:72b",
        format="json",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
    )

    # 检查并解析响应
    content = response['message']['content']
    print(f"response content:\n{content}")

    try:
        result = json.loads(content)
    except json.JSONDecodeError as e:
        print("Failed to decode JSON response:", e)
        result = {"flag": False, "detail": "JSON解析失败", "cpe": []}

    return result


# 处理每个文件并生成结果
for file_name in output_files:
    with open(os.path.join(input_dir, file_name), 'r', encoding='utf-8') as f:
        content = f.read()

    vulnerability_intelligence, ai_output = extract_vulnerability_info(content)

    if vulnerability_intelligence and ai_output:
        result = query_qwen_72b(vulnerability_intelligence, ai_output)
        print(f"File: {file_name}")
        pprint(result)
    else:
        print(f"Error extracting data from {file_name}")
