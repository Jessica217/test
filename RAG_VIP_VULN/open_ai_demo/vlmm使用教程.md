以下是启动 vLLM 的 OpenAI 兼容 API 服务的详细指南，包括常用选项的解释和如何根据需求构建命令行参数。

### 1. 基本启动参数

1. **模型路径或名称**：**必选**，用于指定要部署的模型，可以是 Hugging Face 的模型名称或本地路径。
   - 示例：`/data/huggingface_models/Qwen2.5-14B-Instruct`

2. **`--host`** 和 **`--port`**：用于设置 API 服务的主机名和端口号。
   - 示例：`--host 0.0.0.0 --port 8080`

3. **`--api-key`**：指定 API 密钥，客户端必须在请求头中提供该密钥以进行身份验证。
   - 示例：`--api-key my_secret_key`

### 2. 性能优化选项

1. **`--gpu-memory-utilization`**：设置 GPU 内存的使用比例，默认是 0.9；可通过调整该值避免内存溢出。
   - 示例：`--gpu-memory-utilization 0.8`

2. **`--max-model-len`**：指定模型的最大上下文长度。可以减小此值以减少内存占用。
   - 示例：`--max-model-len 10240`

3. **`--tensor-parallel-size`**：指定张量并行数量，用于多 GPU 分布式部署。
   - 示例：`--tensor-parallel-size 2`

### 3. JSON 输出配置选项

1. **`--guided-decoding-backend`**：设置引导解码的后端，确保输出遵循指定格式。可选值包括 `outlines` 和 `lm-format-enforcer`。
   
   - 示例：`--guided-decoding-backend outlines`
   
2. **`--guided-json`**：定义输出的 JSON 模式，以确保响应遵循特定结构。
   - 示例：`--guided-json '{"type": "object", "properties": {"response": {"type": "string"}, "metadata": {"type": "object"}}, "required": ["response"]}'`

   **注意**：`--guided-json` 是一个请求级别的参数，需在每个请求中单独指定，而非在启动服务时全局设置。

### 4. 调试和日志选项

1. **`--uvicorn-log-level`**：设置 Uvicorn 的日志级别，如 `info`、`debug` 等。
   - 示例：`--uvicorn-log-level info`

2. **`--max-log-len`**：定义日志中打印的最大字符数，以帮助控制日志文件大小。
   - 示例：`--max-log-len 100`

### 示例启动命令

以下命令将 Qwen 模型作为 OpenAI 兼容的 API 服务部署在本地，支持 JSON 格式输出并限制 GPU 内存使用：

```bash
vllm serve /data/huggingface_models/Qwen2.5-14B-Instruct \
  --host 0.0.0.0 \
  --port 8080 \
  --gpu-memory-utilization 0.8 \
  --max-model-len 10240 \
  --tensor-parallel-size 2 \
  --guided-decoding-backend outlines \
  --uvicorn-log-level info \
  --served-model-name Qwen/Qwen2.5-14B-Instruct \
  --dtype bfloat16 \
  > vllm_logfile.log 2>&1

```

**注意**：`--guided-json` 参数应在每个请求中指定，而非在启动命令中全局设置。你在哪个目录，就会把模型部署到哪个目录，所以是在data/wjy下。

### 请求示例

要生成符合指定 JSON 模式的响应，请在发送请求时包含 `guided_json` 和 `guided_decoding_backend` 参数。例如，可以使用 Python 的 `openai` 库发送请求：

```python
from openai import OpenAI
def get_prompt(vip_vuln_info):
    prompt = f"""You are a world-class cybersecurity expert specializing in the analysis and handling of vulnerability intelligence. From the provided vulnerability description, please extract all relevant network security CPEs (Common Platform Enumeration) that adhere to the CPE format. 

    **Requirements for Extraction:** 
    1. The output should consist only of strings in the exact CPE format: `cpe:/{{part}}:{{vendor}}:{{product}}:{{version}}:{{update}}:{{edition}}:{{language}}`.
    2. If certain fields are missing in the original CPE data, retain the details as they are without adding or modifying any information.

    **Output Specifications:** 
    - Format the results as a JSON object with a single key, `cpe`, whose value is an array listing each extracted CPE string individually.
    - No extra explanations or descriptions are required.

    ### Vulnerability Intelligence:
    {vip_vuln_info}
    """
    return prompt

# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8080/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

model = "Qwen/Qwen2.5-14B-Instruct"

vip_vuln_info="""VPL-JAIL-SYSTEM是jcrodriguez-dis个人开发者的一个库。为 VPL Moodle 插件提供了一个执行沙盒。
VPL-JAIL-SYSTEM v4.0.2 版本及之前版本存在安全漏洞，该漏洞源于存在路径遍历问题。
输出：漏洞影响的产品：VPL-JAIL-SYSTEM，漏洞影响的版本：<=v4.0.2
"""

prompt = get_prompt(vip_vuln_info)

chat_response = client.chat.completions.create(
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
print("Chat response:", chat_response)

content=chat_response.choices[0].message.content
print(f"content:\n{content}")
```

在上述代码中，`extra_body` 字段包含 `guided_json` 和 `guided_decoding_backend` 参数，用于定义期望的 JSON 输出结构和解码后端。

### 关于 `--guided-decoding-backend outlines` 参数

**`--guided-decoding-backend outlines`** 参数指定了引导解码的后端。`outlines` 是一个支持强制格式化输出的开源解码后端，例如确保输出内容符合 JSON 模式或其他特定格式。

#### 使用场景

该功能特别适合生成结构化数据（如 JSON 格式的输出），可以确保模型的输出符合预期的格式，减少数据清理和手动调整的需求。

#### 请求示例

在每个请求中指定 `guided_json` 和 `guided_decoding_backend` 参数来生成符合 JSON 结构的响应。

```json
{
  "model": "Qwen2.5-14B-Instruct",
  "messages": [
    {"role": "system", "content": "你是一个专业助手"},
    {"role": "user", "content": "请提供一个包含 response 和 metadata 的 JSON 响应"}
  ],
  "guided_json": {
    "type": "object",
    "properties": {
      "response": {"type": "string"},
      "metadata": {"type": "object"}
    },
    "required": ["response"]
  }
}
```

在这种配置下，模型会尝试生成符合指定 JSON 结构的响应。

### 参考资料

- [vLLM 官方文档](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)
- [Outlines GitHub 仓库](https://github.com/dottxt-ai/outlines) 

通过以上配置，可以高效启动 vLLM 服务，并实现符合 JSON 格式的输出。

---

