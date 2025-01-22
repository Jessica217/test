import gradio as gr
import pandas as pd
from pymilvus import connections, Collection
from milvus_model.hybrid import BGEM3EmbeddingFunction
from ollama import Client
import re
from openai import OpenAI
import json
from pydantic import BaseModel, Field

#连接Qwen_Max
import os
# DASHSCOPE_API_KEY='sk-75bb6015f0d249408148a2bc37620525' #写到config文件中

# DASHSCOPE_API_KEY='40fb89ca21222f4bcef79d814d2dfeab.CPDlP2b8CL2W0qeb' #写到config文件中
DASHSCOPE_API_KEY='sk-KJziv58j8TxW2j895f53B46408E04311Ac4121FcCa8f98Eb' #写到config文件中
client = OpenAI(
    api_key= DASHSCOPE_API_KEY,
    # base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    # base_url = "https://open.bigmodel.cn/api/paas/v4/",
    base_url = "http://43.154.251.242:10050/v1",
)

# 加载嵌入模型
model_name_or_path = '/data/huggingface_models/bge-m3'
bge_m3_ef = BGEM3EmbeddingFunction(
    model_name=model_name_or_path,
    device='cuda:0', # 不使用gpu
    use_fp16=False
)

# 连接到 Milvus向量数据库
connections.connect(alias="default", host="10.0.81.173", port="19530")
collection_name = "wjy_new_demo3"
collection = Collection(name=collection_name)

class CPESearchResult(BaseModel):
    index: int
    part: str
    vendor: str
    product: str
    version: str
    update: str
    edition: str
    language: str
    score: float

# # 配置 Ollama 的 LLM 环境变量
# url = 'http://10.0.81.173:11434'
# client = Client(host=url)


# 生成用户消息 Prompt
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


# 非流式输出函数，使用 Qwen MAX
def chat_with_qwen_max(client, model_name, user_message):
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_message},
        ],
        temperature = 0.5,
        top_p = 0.8,
        max_tokens = 8192
    )
    content = completion.choices[0].message.content
    return content

def extract_all_cpe_strings(content):
    try:
        # 解析 JSON 内容
        content_dict = json.loads(content)
        # 直接从字典中获取 'cpe' 键的值，如果不存在就返回 None
        return content_dict.get("cpe", None)
    except json.JSONDecodeError:
        # 如果 JSON 格式错误，返回 None
        return None

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

# 搜索 CPE 数据
def search_cpe(input_text):
    # 生成用户消息并调用大模型
    user_message = generate_prompt(input_text)
    # model_name = 'qwen2.5:72b'
    # model_name = "/data/huggingface_models/Qwen2.5-14B-Instruct"
    #使用qwen_max
    # model_name = 'qwen-max-latest'
    # model_name = 'glm-4-plus'

    model_name = 'gpt-4o-2024-08-06'
    content = chat_with_qwen_max(client, model_name, user_message)
    print('==================================LLM_content====================', content)

    # 提取所有 CPE
    all_cpe = extract_cpe(content)
    # print(type(first_cpe)) str
    if not all_cpe:
        return {"cpe": "No valid CPE extracted."}, pd.DataFrame()
    print(all_cpe)  # 输出 ['cpe:/a:adobe:acrobat:*:*:*:*', 'cpe:/a:adobe:reader:*:*:*:*']

    # 转为 JSON 格式
    cpe_json_formatted = json.dumps({"cpe": all_cpe}, indent=4)

    # 初始化存储所有搜索结果
    all_results = []
    version_filter_result = []

    # 遍历每个 CPE，逐个生成向量并进行搜索
    for cpe_str in all_cpe:
        # 调用修复后的 cpe_to_json 函数
        cpe_json = cpe_to_json(cpe_str)
        if "error" in cpe_json:
            print(f"Skipping invalid CPE: {cpe_json['error']}")
            continue

        # 确保 JSON 格式正确
        print("Processing CPE:", cpe_json)
        # 使用嵌入模型生成查询向量
        try:
            doc_embedding = bge_m3_ef.encode_queries([json.dumps(cpe_json)])
            dense_vector = doc_embedding['dense'][0]
        except Exception as e:
            print(f"Error in embedding generation: {e}")
            continue

        # 搜索 Milvus 数据库

        search_params = {
            "metric_type": "COSINE",
            "params": {'nprobe': 20, 'level': 3, 'radius': 0.8, 'range_filter': 1}
        }
        results = collection.search(
            data=[dense_vector], anns_field="vector", param=search_params,
            limit=50, output_fields=["part", "vendor", "product", "version", "update", "edition", "language"]
        )

        # 格式化当前 CPE 的搜索结果
        index = 1
        for hits in results:
            for hit in hits:
                row = {
                    "index": index,
                    # "cpe": cpe_str,  # 当前搜索的 CPE
                    "part": hit.entity.get("part"),
                    "vendor": hit.entity.get("vendor"),
                    "product": hit.entity.get("product"),
                    "version": hit.entity.get("version"),
                    "update": hit.entity.get("update"),
                    "edition": hit.entity.get("edition"),
                    "language": hit.entity.get("language"),
                    "score": round(hit.score * 100, 2)
                }
                all_results.append(row)
                index += 1
            # 如果没有搜索结果
        if not all_results:
            return {"cpe": "Valid CPEs found, but no results from Milvus search."}, pd.DataFrame(), pd.DataFrame()


        for idx, hit in enumerate(results[0], start=1):
            score = round(hit.score * 100, 2)
            version_filter_result.append(CPESearchResult(
                index=idx,
                part=hit.entity.get("part"),
                vendor=hit.entity.get("vendor"),
                product=hit.entity.get("product"),
                version=hit.entity.get("version"),
                update=hit.entity.get("update"),
                edition=hit.entity.get("edition"),
                language=hit.entity.get("language"),
                score=score
            ))

    sorted_data = sorted(version_filter_result, key=lambda x: version_key(x.version))  # 按照版本号排序
    json_list = [item.dict() for item in sorted_data]  # 转为字典列表
    print(f"sorted_json_list:{json_list}")
    #
    reuse_prompt = filter_prompt(input_text, json_list)
    print(f"reuse_prompt:{reuse_prompt}")
    content = chat_with_qwen_max(client, model_name, reuse_prompt)
    print(f"llm_content:{content}")
    #
    extracted_list = extract_list_from_content(content)
    print(f"extracted_list:{extracted_list}")
    version_extract_list = pd.DataFrame(extracted_list)

        # 转换为 DataFrame
    df = pd.DataFrame(all_results)
    return cpe_json_formatted, df, version_extract_list
    # return cpe_json_formatted, df




# 创建 Gradio 界面
with gr.Blocks() as demo:
    with gr.Column():
        user_input = gr.Textbox(label="输入框：请输入vip漏洞情报内容")
        submit_button = gr.Button("提交")
        output_data1 = gr.Textbox(label="大模型提取的 CPE 数据（JSON 格式）", interactive=False)
        output_data2 = gr.Dataframe(label='检索Milvus向量数据并返回相似的CPE数据', interactive=False)
        output_data3 = gr.Dataframe(label='再次使用大模型筛选符合版本号条件的CPE版本号', interactive=False)

    # submit_button.click(fn=search_cpe, inputs=user_input, outputs=[output_data1, output_data2])
    submit_button.click(fn=search_cpe, inputs=user_input, outputs=[output_data1, output_data2, output_data3])
def version_key(version):
  # 定义字母的排序顺序，字母按照某个优先级排序
  alpha_order = {'a': -1, 'b': 0, 'rc': 1, '': 2}  # 空字符串为空版本，优先级最后
  # 将版本号拆分为多个部分
  version_parts = version.split('.')
  # 用于存储拆解后的版本号部分
  version_parts_as_numbers = []
  for part in version_parts:
    # 处理版本号部分，如果部分是纯数字，则转换为整数
    if part.isdigit():
      version_parts_as_numbers.append(int(part))
    else:
      # 如果是带字母的版本号（例如 1.0.0b1），需要将字母映射到排序规则
      number_part = ''.join([char for char in part if char.isdigit()])
      letter_part = ''.join([char for char in part if not char.isdigit()])
      # 只有在数字部分不为空时才转换为整数
      if number_part:
          version_parts_as_numbers.append(int(number_part))
      else:
          version_parts_as_numbers.append(0)  # 或者你可以选择一个默认值，例如 0
      # 将字母部分转换为对应的优先级
      version_parts_as_numbers.append(alpha_order.get(letter_part, 2))  # 默认字母为空字符串

  return version_parts_as_numbers
def filter_prompt(vip_vuln_info, rerank_data):
    prompt = f"""
        基于以下漏洞描述和CPE匹配结果，筛选出满足漏洞描述中版本限制条件的CPE信息，返回一个过滤后的结果。注意：返回内容必须是纯 Python 列表格式，且不能包含任何其他文字或说明。**
    ### 处理要求：

    1. **提取版本限制条件**  
   - 从漏洞描述中识别并提取版本限制条件（包括符号如 `<`、`<=`、`>=`、`>`，或文字如“更早”、“之前”、“至”等，以及对应的版本号）。  
   - 如果漏洞描述未明确提供限制条件，则默认版本限制条件为 `=`，即版本必须与描述中的版本号完全一致。
   - 特别注意：如果版本号中包含类似“LibreCAD LibreCAD <=2.2.0-rc3”这类形式，其中版本号后带有“-”符号（如 rc、beta、alpha 等），则应将版本限制条件视为 <=（小于等于），而非仅限于等于 =。
     示例，LibreCAD <=2.2.0-rc3 应被解释为“版本号小于等于 2.2.0-rc3”，并不会被当做“仅等于 2.2.0-rc3”进行匹配。

    2. **筛选 CPE 匹配结果**  
   - 依据提取的版本限制条件，筛选提供的 CPE 列表，仅保留满足条件的 CPE 信息。  
   - 不符合条件的 CPE 条目必须完全排除。

    3. **输出格式**  
   - 返回筛选后的 CPE 列表，保留原始数据的字段结构，包括 `index`、`part`、`vendor`、`product`、`version`、`score`。  
   - 如果没有符合条件的 CPE，返回一个空列表。  
   - 输出内容必须为**纯 Python 列表**格式，不得包含任何其他文字。

    ### 漏洞描述：  
    {vip_vuln_info}
    ### CPE匹配结果（基于version字段升序排序的Python列表）： 
    {rerank_data}
    ### 输出要求：  
    仅返回一个格式为纯 Python 列表的JSON数组，包含所有满足条件的CPE信息，其他不符合版本号的CPE不要返回！！如果漏洞描述中未提供明确的版本限制条件，则返回一个空列表。

"""
    return prompt
def extract_list_from_content(content):
    # 使用正则表达式匹配Python代码块中的列表部分
    match = re.search(r'```python\n(.*?)\n```', content, re.DOTALL)
    if match:
        # 提取匹配到的内容并解析为Python列表
        list_str = match.group(1)
        print(f"list_str: {list_str}")
        new_list_str = list_str.replace("'", '"')
        # 打印替换后的字符串进行调试
        print(f"Modified list_str: {list_str}")  # 依然是单引号
        import ast
        try:
            extracted_list = ast.literal_eval(new_list_str)
            return extracted_list
        except (ValueError, SyntaxError) as e:
            print(f"Error parsing list: {e}")
            return []
    else:
        return []

# 启动界面
demo.launch(server_name='0.0.0.0', server_port=58085)
