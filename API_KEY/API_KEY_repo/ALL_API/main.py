from fastapi import FastAPI, HTTPException, Response, Query
from pydantic import BaseModel, Field
from typing import List, Optional
from pymilvus import connections, Collection, MilvusClient
import json
import configparser
from milvus_model.hybrid import BGEM3EmbeddingFunction
from openai import OpenAI
import re
import string
import requests
import numpy

import tiktoken


# 读取配置文件
config = configparser.ConfigParser()
config.read('config.ini')
app = FastAPI()

#连接到Milvus数据库
connections.connect("default", host=config['MILVUS']['Host'], port=config['MILVUS']['Port'])
collection = Collection(config['MILVUS']['CollectionName'])

# 连接到 Qwen_Max 使用配置中的 API Key
client = OpenAI(
    api_key=config['OPENAI']['GLM4_API_KEY'],
    # api_key=config['OPENAI']['deepseek_API_KEY'],
    #base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", # qwen url
    base_url="https://open.bigmodel.cn/api/paas/v4/", # glm4 url
    # base_url="https://api.deepseek/v1", # deep seek url


)

bge_m3_ef = BGEM3EmbeddingFunction(
        model_name_or_path=config['MODEL']['ModelPath'],
        device=config['MODEL']['Device'],
        #device = 'auto',
        use_fp16=config['MODEL']['UseFP16'].lower() == 'true'
    )

#使用vulnerabilityInfo 方法，使得输入可以是json格式的。
class VulnerabilityInfo(BaseModel):
    vuln_name: str
    vuln_desc: str
    effect_scope: Optional[str] = None



class Tokenizer:
    """ required: import tiktoken; import re;
    usage example:
        cl100 = Tokenizer()
        number_of_tokens = cl100.count("my string")
    """

    def __init__(self, model="cl100k_base"):
        self.tokenizer = tiktoken.get_encoding(model)
        self.chat_strip_match = re.compile(r'<\|.*?\|>')
        self.intype = None
        self.inprice = 0.01 / 1000  ### hardcoded GPT-4-Turbo prices
        self.outprice = 0.03 / 1000

    def ucount(self, text):
        encoded_text = self.tokenizer.encode(text)
        return len(encoded_text)

    def count(self, text):
        text = self.chat_strip_match.sub('', text)
        encoded_text = self.tokenizer.encode(text)
        return len(encoded_text)

    def outputprice(self, text):
        return self.ucount(text) * self.outprice

    def inputprice(self, text):
        return self.ucount(text) * self.inprice

    def message(self, message):
        if isinstance(message, str):
            self.intype = str
            message = dict(message)
        if isinstance(message, dict):
            self.intype = dict
            message = [message]
        elif isinstance(message, list):
            self.intype = list
        else:
            raise ValueError("no supported format in message")
        for msg in message:
            role_string = msg['role']
            if 'name' in msg:
                role_string += ':' + msg['name']
            role_tokens = self.count(role_string)
            content_tokens = self.count(msg['content'])
            msg['tokens'] = 3 + role_tokens + content_tokens
            msg['price'] = msg['tokens'] * self.inprice
        return message if len(message) > 1 else message[0]
#
# cl100 = Tokenizer()
# number_of_tokens = cl100.count("my string")
# print(number_of_tokens)

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



#输入cpe数据，然后存到数据库中
@app.post("/insert_cpe", description='此接口用于插入新的数据到向量数据库中。')  # 路径操作函数
async def vectorize_cpe(data: list[str]): # 输入的data是list类型的，
    results = []
    # 连接到 Milvus
    connections.connect("default", host=config['MILVUS']['Host'], port=config['MILVUS']['Port'])
    client = MilvusClient(connection="default")
    res = client.query(
        collection_name="wjy_new_demo3",
        output_fields=["count(*)"]
    )
    cpe_count = res[0]['count(*)']  # 访问字典中的值
    # 分批处理数据
    batch_size = 10000
    batches = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]  # 将数据分批
    for batch in batches:
        for cpe_data in batch:
            cpe_count += 1
            cpe_json = cpe_to_json(cpe_data)
            print("Count:", cpe_count)
            collection_name = config['MILVUS']['CollectionName']
            # 若集合不存在，则创建集合
            if collection_name not in client.list_collections():
                client.create_collection(
                    collection_name=collection_name,
                    dimension=int(config['MILVUS']['Dimension'])
                )

            # 加载向量嵌入模型
            bge_m3_ef = BGEM3EmbeddingFunction(
                model_name_or_path=config['MODEL']['ModelPath'],
                device=config['MODEL']['Device'],
                use_fp16=config['MODEL']['UseFP16'].lower() == 'true'
            )

            doc_embedding = bge_m3_ef.encode_documents([json.dumps(cpe_json)])
            dense_vector = doc_embedding["dense"][0]
            data_dict = {
                "id": cpe_count,
                "vector": dense_vector,
                "cpe_json": cpe_json,
                "part": cpe_json.get("part", ""),  # 获取 'part' 字段
                "vendor": cpe_json.get("vendor", ""),  # 获取 'vendor' 字段
                "product": cpe_json.get("product", ""),  # 获取 'product' 字段
                "version": cpe_json.get("version", ""),  # 获取 'version' 字段
                "update": cpe_json.get("update", ""),  # 获取 'update' 字段
                "edition": cpe_json.get("edition", ""),  # 获取 'edition' 字段
                "language": cpe_json.get("language", "")  # 获取 'language' 字段
            }
            target_data = []
            target_data.append(data_dict)
            print(target_data)
            # 将数据插入到 Milvus
            res = client.insert(collection_name=collection_name, data=target_data)
            results.append(res)
            print(f"Insert result for {cpe_data}: {res}")

        # print(f"Insert result: {res}")

    return {"status": "OK", "message": "CPE data processed and vectorized successfully"}

@app.post("/extract_cpe",description='此接口使用大模型进行对漏洞情报中CPE字段的提取。')
async def extract_cpe(vip_vuln_info: VulnerabilityInfo):
    ###统计输入的token
    print('====================================input_token==========================')
    count_tokens = Tokenizer()
    json_string = str(vip_vuln_info)
    number_of_input_tokens = count_tokens.count(json_string)
    # print(f'number_of_input_tokens,{number_of_input_tokens}')
    # response.body["Token-input"] = str(number_of_input_tokens)


    """从漏洞描述中提取CPE字符串的API端点。"""
    prompt = get_prompt_no_detail(vip_vuln_info.model_dump_json())
    print('====================================user_message==========================')
    print(prompt)
    # content = await chat_with_qwen_max(client, 'qwen-max-latest', prompt)
    content = await chat_with_qwen_max(client, 'glm-4-plus', prompt)
    #content = await chat_with_qwen_max(client, 'glm', prompt)

    print('====================================content==========================')
    print(content)

    print('====================================input_token==========================')
    number_of_output_tokens = count_tokens.count(content)
    print(f'number_of_output_tokens,{number_of_output_tokens}')

    total_token = number_of_input_tokens + number_of_output_tokens

    token_detail_info = {
        "input_token": number_of_input_tokens,
        "output_token": number_of_output_tokens,
        "total_token": total_token,
        "提示":"下面费用仅供参考",
        "input_price": count_tokens.inputprice(json_string),
        "output_price": count_tokens.outputprice(content),
        #"total_price": count_tokens.totalprice(total_token),
    }


    cpe_list = process_extracted_cpe(content)

    # 输出json格式的cpe字符
    all_cpe_list = []
    for cpe_str in cpe_list:
        # 调用修复后的 cpe_to_json 函数
        cpe_json = cpe_to_json(cpe_str)
        all_cpe_list.append(cpe_json)

    return {"cep_json": all_cpe_list, "cpe_list":cpe_list, "token_detail_info":token_detail_info}


# @app.post("/search_cpe", response_model=List[CPESearchResult])
@app.post("/search_cpe",description='此接口用于返回符合向量数据库中符合漏洞描述的CPE字段，若数据库中没有即[],则返回上一步由大模型提取的CPE字段。'
                                    '用户需要通过输入大模型名称，base_url和api_key三个参数，完成对大模型的选择。')
async def search_cpe(vip_vuln_info: VulnerabilityInfo,
                     #cpe_list: List[str],
                     # model_name: str = Query(..., enum=["gpt-4o-2024-08-06", "qwen-max-latest", "glm-4-plus"、“deepseek-chat], description="gpt-4"), # 选择模型
                     model_name: str = Query(..., description="LLM_model_name"), # 选择模型
                     base_url:str = Query(..., description="API base URL"),
                     api_key:str = Query(..., description="API key")
                     ):
        all_results = []
        cpe_data = await extract_cpe(vip_vuln_info)
        # 从 extract_cpe 的返回结果中提取 cpe_list
        cpe_list = cpe_data["cpe_list"]
        for cpe_str in cpe_list:
            cpe_json = cpe_to_json(cpe_str)
            doc_embedding = bge_m3_ef.encode_queries([json.dumps(cpe_json)])
            dense_vector = doc_embedding['dense'][0]
            print(f"dense_vector:{dense_vector}")
            search_params = {
                "metric_type": "COSINE",
                "params": {'nprobe': 20, 'level': 3, 'radius': 0.8, 'range_filter': 1} #radius是返回余弦相似度分数在0.8以内的搜索结果
            }
            results = collection.search(
                data=[dense_vector], anns_field="vector", param=search_params,
                limit=50, output_fields=["part", "vendor", "product", "version", "update", "edition", "language"]
            )
        # Append search results
            for idx, hit in enumerate(results[0], start=1):
                score = round(hit.score * 100, 2)
                all_results.append(CPESearchResult(
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

        sorted_data = sorted(all_results, key=lambda x: version_key(x.version)) # 按照版本号排序
        print('====================================sorted_data==========================')
        print(f"sorted_data:{sorted_data}")
        # 转换为 JSON 列表
        json_list = [item.dict() for item in sorted_data]  # 转为字典列表
        print(f"sorted_json_list:{json_list}")

        reuse_prompt = filter_prompt(vip_vuln_info, json_list) # 筛选提示
        print('===================================reuse_prompt==================================')
        print(f"reuse_prompt:{reuse_prompt}")

        print('====================================content==========================')
        # content = await reuse_chat_with_qwen_max(client, 'glm-4-plus',reuse_prompt) # 提取提示
        # content = reuse_chat_with_model(client, 'gpt-4o-2024-08-06', reuse_prompt)
        # content = reuse_chat_with_model(client, model_name, reuse_prompt)
        content = reuse_chat_with_model(base_url, api_key, model_name, reuse_prompt)

        print(f"llm_content:{content}")

        extracted_list = extract_list_from_content(content)
        print(f"extracted_list:{extracted_list}")  # 其中Json已经是双引号 ”“
        # 判断 extracted_list 是否为空
        if extracted_list == []:
            return {"result":cpe_list, "source": "llm_extract_cpe"}
        else:
            return {"result":extracted_list, "source": "vector_database"}


def reuse_chat_with_model(base_url, api_key, model_name, user_message):
    """根据模型名称选择不同的生成模型"""
    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )
    response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_message},
            ],
            temperature=0.5,
            top_p=0.6,
            max_tokens=8192,
        )
    return response.choices[0].message.content

def get_prompt_no_detail(vip_vuln_info):
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

#### Input Example 3:

```json
{{
  "vuln_name": "Industrial control software vulnerability",
  "vuln_desc": "Privilege escalation vulnerabilities exist in Mitsubishi Electric's MC Works64 and Iconics Genesis64 software, affecting versions <=2.4.6.",
  "effect_scope": "Industrial control software"
}}
```

#### Output Example 3:

```json
{{
  "cpe": [
    "cpe:/a:mitsubishi_electric:mc_works64:2.4.6:*:*:*",
    "cpe:/a:iconics:genesis64:2.4.6:*:*:*"
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

###定义对于检索top50之后的内容的升序排序后，从此结果中提取出满足版本限制条件的CPE版本信息,做RAG，其中提示词中包含上一步大模型提取出来的CPE字段，然后再根据新的提示词，筛选出对应的符合版本号的cpe。
def filter_prompt(vip_vuln_info, rerank_data):
    prompt = f"""基于以下漏洞描述和CPE匹配结果，筛选出满足漏洞描述中版本限制条件的CPE信息，返回一个过滤后的结果。注意：返回内容必须是纯 Python 列表格式，且不能包含任何其他文字或说明。**
    ### 处理要求：

    1. **提取版本限制条件**  
   - 从漏洞描述中识别并提取版本限制条件（包括符号如 `<`、`<=`、`>=`、`>`，或文字如“更早”、“之前”、“至”等，以及对应的版本号）。  
   - 如果漏洞描述未明确提供限制条件，则默认版本限制条件为 `=`，即版本必须与描述中的版本号完全一致。

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

async def reuse_chat_with_qwen_max(client, model_name,user_message):
    """与OpenAI的Qwen Max模型交互，获取CPE提取结果。"""
    completion =  client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_message},
        ],
        temperature=0.5,
        # top_p=0.8,
        top_p=0.6,
        max_tokens=8192,
        extra_body={"repetition_penalty": 1.05}
    )
    return completion.choices[0].message.content

#######################################################################QWen-MAX###########################################################
async def chat_with_qwen_max(client, model_name, user_message):
    """与OpenAI的Qwen Max模型交互，获取CPE提取结果。"""
    completion =  client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_message},
        ],
        temperature=0.5,
        # top_p=0.8,
        top_p=0.6,
        max_tokens=8192,
        extra_body={"repetition_penalty": 1.05}
    )
    return completion.choices[0].message.content
#######################################################################GLM4###########################################################
def chat_with_model_non_stream(client, model_name, user_message, system_message="你是一个人工智能助手，你叫chatGLM", temperature=0.5, max_tokens=8192):
    #start = datetime.datetime.now()
    print(f"user_message:{user_message}")
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
        temperature=0.5,
        top_p=0.8,
        max_tokens=512,
        extra_body={"temperature": temperature, "max_tokens": max_tokens},
    )

    content = response.choices[0].message.content
    print(f"content:{content}")
    #elapsed_time = (datetime.datetime.now() - start).total_seconds()
    return content

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

def cpe_to_json(cpe_str):
    fields = ["part", "vendor", "product", "version", "update", "edition", "language"]
    if not cpe_str.startswith("cpe:/"):
        return {"error": f"Invalid CPE format: {cpe_str}"}
    parts = cpe_str.replace("cpe:/", "").split(":")
    if len(parts) > len(fields):
        return {"error": f"Invalid CPE format: {cpe_str} (too many fields)"}
    cpe_json = {fields[i]: parts[i] if i < len(parts) else "*" for i in range(len(fields))}
    return cpe_json


import ast
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
        try:
            extracted_list = ast.literal_eval(new_list_str)
            return extracted_list
        except (ValueError, SyntaxError) as e:
            print(f"Error parsing list: {e}")
            return []
    else:
        return []
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8011)
