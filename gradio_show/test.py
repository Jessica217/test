import gradio as gr
import json
import re
from ollama import Client

# 配置 Ollama 的 LLM 环境变量
url = 'http://10.0.81.173:11434'
client = Client(host=url)

# 生成用户消息 Prompt
def get_prompt(vip_vuln_text):
    user_message = f"""
    You are a world-class cybersecurity expert specializing in the analysis and processing of vulnerability intelligence. Please extract all valid **CPE (Common Platform Enumeration)** strings from the following vulnerability description. Extraction rules and output format are as follows:

---

### Extraction Rules

1. **Strict Adherence to the CPE Standard Format:**  
   CPE strings must strictly follow the format:  cpe:/{{part}}:{{vendor}}:{{product}}:{{version}}:{{update}}:{{edition}}:{{language}}， A total of 7 fields must be included. If any field is missing, it must be replaced with the placeholder `*`, ensuring all fields are complete.

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
{vip_vuln_text}
    """
    return user_message

# 使用 Qwen 72B
def chat_with_qwen_72b(client, model_name, user_message):
    """调用大模型并返回输出"""
    response = client.chat(
        model=model_name,
        format="json",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_message},
        ]
    )
    return response['message']['content']

# 从内容中提取第一个 CPE 字符串
def extract_first_cpe_string(content):
    try:
        # 将 JSON 字符串解析为 Python 字典
        content_dict = json.loads(content)

        # 获取 CPE 列表
        cpe_list = content_dict.get("cpe", [])
        if not cpe_list:
            return "No CPEs found in the content."

        # 匹配第一个 CPE 字符串
        pattern_cpe = r"cpe:\/([ahoil]):([\w-]+):([\w-]+|\*):([<>=]*[\w\.-]*|\*):([\w\.-]*|\*):([\w\.-]*|\*):([\w\.-]*|\*)"
        for cpe in cpe_list:
            match = re.match(pattern_cpe, cpe)
            if match:
                return 'cpe:/{}:{}:{}:{}:{}:{}:{}'.format(*match.groups())

        return "No valid CPEs matched the pattern."
    except Exception as e:
        return f"Error extracting CPE: {e}"

def cpe_to_json(cpe_str):
    # 去掉前缀 "cpe:/"
    parts = cpe_str.replace("cpe:/", "").split(":")

    # CPE字段映射
    fields = ["part", "vendor", "product", "version", "update", "edition", "language"]

    # 将CPE字段与内容对应
    cpe_json = {fields[i]: parts[i] if i < len(parts) else "" for i in range(len(fields))}

    # 返回JSON格式
    return json.dumps(cpe_json, indent=4)

# 主处理函数：整合提取与正则逻辑
def handle_input(vip_vuln_text):
    # 生成用户消息
    user_message = get_prompt(vip_vuln_text)

    # 调用大模型并获取结果
    model_name = 'qwen2.5:72b'
    content = chat_with_qwen_72b(client, model_name, user_message)
    # # 尝试将内容解析为 JSON 美化格式
    # content_dict = json.loads(content)
    # formatted_content = json.dumps(content_dict, indent=4, ensure_ascii=False)
    # 尝试将内容解析为 JSON 并美化输出
    # try:
    #     content_dict = json.loads(content)  # 确保 content 是 JSON 格式
    #     formatted_content = json.dumps(content_dict, indent=4, ensure_ascii=False)  # 美化 JSON 为多行
    # except json.JSONDecodeError:
    #     # 如果不是 JSON 格式，直接替换转义换行符为实际换行符
    #     formatted_content = content.replace("\\n", "\n")

    # 提取第一个 CPE
    first_cpe = extract_first_cpe_string(content)
    json_cpe = cpe_to_json(first_cpe)



    # 返回大模型提取的原始内容和正则提取的第一个 CPE
    return content, first_cpe, json_cpe

# # 创建 Gradio 界面
# with gr.Blocks() as demo:
#     with gr.Column():
#         user_input = gr.Textbox(label="请输入VIP漏洞情报内容", placeholder="请输入包含漏洞描述的文本...", lines=5)
#         output_full = gr.Textbox(label="大模型提取的完整CPE内容", lines=10, interactive=True)
#         output_first = gr.Textbox(label="正则提取的第一个CPE", lines=1, interactive=False)
#         submit_button = gr.Button("提交")
#
#     submit_button.click(fn=handle_input, inputs=user_input, outputs=[output_full])
#
# # 启动界面
# demo.launch(server_name='0.0.0.0', server_port=58082)