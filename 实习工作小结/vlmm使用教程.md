### 一、以下是启动 vLLM 的 OpenAI 兼容 API 服务的详细指南，包括常用选项的解释和如何根据需求构建命令行参数。

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

### 二、关于使用**vllm部署在服务器上** 和 **调用服务器本地模型**的区别：

是的，**将模型部署在服务器上** 和 **直接使用服务器中本地的模型** 是两回事，它们在实现、管理和使用上有一些关键的区别。

### 1. **将模型部署在服务器上（模型部署）**

模型部署通常意味着将已经训练好的模型放置在一个专门的服务端，并通过 API 或其他形式提供访问接口。用户或应用程序可以通过网络请求与该服务进行交互，获取模型的预测结果。

#### 特点：

- **远程访问**：用户通过网络（例如 HTTP 请求）调用模型服务，而不是直接访问本地模型文件。
- **易于更新和维护**：模型可以在服务器上随时更新，避免了在每个客户端设备上进行手动更新。
- **可扩展性**：通过部署在服务器上，可以更方便地进行负载均衡，处理更多的并发请求，支持大规模使用。
- **集中管理**：服务器可以集成日志、监控、性能优化等管理工具，更方便地管理模型的运行和版本。
- **服务化**：模型通常作为一个 API 服务提供，比如你可以使用 FastAPI 或 Flask 来创建一个 RESTful API 供用户请求。

#### 优点：

- **集中管理**：服务器端集中式管理，便于监控和维护。
- **多用户共享**：多个用户可以同时通过 API 调用模型，避免重复部署和资源浪费。
- **跨平台支持**：不依赖于客户端的硬件和操作系统，可以在任何支持网络的设备上进行访问。

#### 缺点：

- **依赖网络**：需要稳定的网络连接来进行远程访问。
- **延迟问题**：网络延迟可能会影响到模型响应时间，特别是在需要快速响应的场景中。
- **资源消耗**：服务器需要足够的计算资源来支撑多个用户的请求，可能需要更多的硬件资源。

### 2. **直接使用服务器中的本地模型**

直接使用本地模型意味着模型已经存在于本地服务器上，应用程序或服务可以直接加载并调用模型进行推理，而不需要通过网络请求。

#### 特点：

- **本地调用**：应用直接加载本地模型文件并执行推理任务，通常不需要依赖外部网络。
- **资源占用**：使用本地模型时，计算资源（CPU、GPU）和内存通常会占用在本地机器上。
- **简化操作**：不需要额外的网络配置，模型调用和推理直接在本地进行。

#### 优点：

- **低延迟**：因为所有操作都在本地进行，不需要经过网络传输，响应时间可以更快。
- **无网络依赖**：适用于没有稳定网络连接的场景，尤其是在内网或离线环境下使用。
- **节省带宽**：不需要频繁地通过网络上传输大量的数据和模型。

#### 缺点：

- **难以扩展**：模型仅在本地服务器上运行，如果需要处理更多请求，需要增加服务器的计算资源（如增加 CPU、GPU）。
- **更新复杂**：每次更新模型时，可能需要在每个使用模型的服务器上进行操作，增加了管理成本。
- **难以共享**：无法像通过 API 一样方便地共享给多个用户或应用。

### 总结

- **模型部署**：适用于需要通过多个客户端共享模型服务的情况，支持跨平台访问，便于更新和维护。
- **本地模型使用**：适用于没有网络限制的环境，响应速度快，适合本地环境下的小规模应用。

### 示例：

- **模型部署**：比如你通过 **FastAPI** 将一个大语言模型部署到云服务器上，用户通过 HTTP 请求调用模型接口，获取模型的预测结果。
- **本地模型**：你将一个经过训练的模型放在本地服务器上，应用直接通过文件路径加载模型并执行推理。

在实际使用中，很多情况下会选择部署模型到服务器上进行集中管理，这样可以处理更多的请求，进行更好的资源分配和监控。不过在一些性能要求非常高或者网络不稳定的情况下，也可能选择本地使用模型。

#### 三、FastAPI和vLLM的 区别

#### **FastAPI** 和 **vLLM** 是两个不同的工具或框架，它们在功能和用途上有显著的区别。下面是它们的详细对比：

### 1. **FastAPI**

**FastAPI** 是一个现代的、高性能的Web框架，专为构建API而设计，特别适合与机器学习模型进行集成，支持高效的异步编程。它基于Python语言构建，使用 **Pydantic** 进行数据验证和 **Starlette** 作为Web框架底层库。

#### 主要特点：

- **Web API 构建**：FastAPI 主要用于构建和提供 API 接口，用户可以通过 HTTP 请求与后端系统交互。
- **异步支持**：内建异步支持（基于 `async` / `await`），适合高并发的应用场景。
- **自动生成文档**：支持自动生成 OpenAPI 和 Swagger 文档，便于开发人员理解和使用 API。
- **易于与 ML 模型集成**：常用于部署机器学习模型为 Web 服务接口，提供推理服务。
- **数据验证**：利用 Pydantic 库，FastAPI 会对传入的数据进行自动验证和解析，确保接口接收到的输入数据符合要求。

#### 适用场景：

- 构建 Web API，特别是针对机器学习模型的接口服务。
- 需要快速创建、扩展和维护的 API 服务。
- 高并发、大量请求的 API 服务器，FastAPI 性能非常优秀。
- 提供 RESTful API 接口，用户可以通过 HTTP 请求获取模型推理结果。

#### 示例：

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/predict")
async def predict(input: str):
    # 这里可以加载模型并进行预测
    result = your_model_predict_function(input)
    return {"prediction": result}
```

------

### 2. **vLLM (vLLM-Server)**

**vLLM** 是一个专门用于部署和运行大规模语言模型（LLM）的优化工具，它特别设计来提高大语言模型的推理效率和资源管理。它支持高效的内存使用、GPU并行化和模型加速，适用于需要大规模并发推理请求的场景。

#### 主要特点：

- **专注于大语言模型推理**：vLLM 主要用于高效地部署和推理大规模的语言模型，如 GPT、BERT、LLAMA 等。它优化了大模型的加载和推理过程，尤其在资源管理和速度上有优势。
- **内存和性能优化**：vLLM 可以在 GPU 上高效运行并进行内存管理，支持多个并发的推理请求，减少计算资源的浪费。
- **支持多种推理后端**：vLLM 支持多种后端（如 FP16、bfloat16、混合精度训练等）来加速推理过程。
- **API 兼容性**：vLLM 支持通过 HTTP API 调用模型服务，并与如 OpenAI API、Hugging Face 服务器兼容。

#### 适用场景：

- 大语言模型的高效推理，尤其是在需要高并发、低延迟的服务场景下。
- 使用大模型（如 GPT-3、LLama、BERT 等）提供推理服务。
- GPU 资源管理和内存优化，适合需要大量 GPU 内存的模型。
- 将模型服务化并通过 HTTP 接口提供给用户，类似 OpenAI API。

#### 示例：

```bash
vllm serve /path/to/your/model \
  --host 0.0.0.0 \
  --port 8080 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 10240
```

该命令启动一个 `vLLM` 服务，监听在 8080 端口，提供模型推理服务。

------

### 对比总结：

| 特性           | **FastAPI**                                            | **vLLM**                                                   |
| -------------- | ------------------------------------------------------ | ---------------------------------------------------------- |
| **主要功能**   | 构建和部署 Web API，提供机器学习模型接口               | 部署和优化大规模语言模型推理服务，提供高效的推理和资源管理 |
| **应用场景**   | 创建快速 API 服务，适合一般 Web 服务、机器学习模型接口 | 专注于大规模语言模型的推理服务，优化 GPU 和内存资源的管理  |
| **主要用途**   | 构建 RESTful API 服务，特别是机器学习模型接口          | 部署大语言模型，提供高并发、高效的推理服务                 |
| **性能优化**   | 支持异步编程，高并发情况下表现优秀                     | 提供 GPU 内存管理和并行化推理，加速大模型推理性能          |
| **API 支持**   | 提供标准 RESTful API，自动生成文档                     | 提供与 OpenAI API 兼容的 HTTP API                          |
| **集成复杂度** | 易于与机器学习模型（如 TensorFlow、PyTorch）集成       | 专注于优化大规模语言模型的推理，集成较为复杂，但性能更强   |
| **常见用例**   | 提供机器学习模型的 API 接口                            | 提供大语言模型推理服务，适用于高并发请求                   |

### 选择指南：

- 如果你需要快速构建一个 **API 服务** 来提供机器学习模型的推理接口，尤其是需要灵活的 API 设计和异步支持，那么 **FastAPI** 是一个合适的选择。
- 如果你正在部署一个 **大规模语言模型**（例如 GPT-3 或其他类似模型），并且需要在高并发和 GPU 资源管理方面有更高效的优化，那么 **vLLM** 是更合适的工具。

**总结**：**FastAPI** 更侧重于构建 Web 服务和 API，而 **vLLM** 更专注于优化大规模语言模型的推理过程，特别是在资源消耗、内存管理和高并发推理方面。
