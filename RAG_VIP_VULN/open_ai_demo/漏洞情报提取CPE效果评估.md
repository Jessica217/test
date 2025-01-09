# 漏洞情报提取CPE效果评估

### 任务提示设计（中文版Prompt）

```plaintext
你是一位顶尖的网络安全专家，专注于分析和处理漏洞情报。根据提供的漏洞描述和大模型的输出，判断模型的输出是否正确，并根据以下要求输出结果。

### 任务要求：
1. **判断输出准确性**：根据原始漏洞描述，核对大模型输出的 CPE 是否准确。
2. **生成正确答案**：如果模型的 CPE 输出不正确，请基于漏洞描述，生成符合 CPE 标准的正确 CPE。

### 判断依据：
- 模型输出的 CPE 必须符合以下格式：`cpe:/{part}:{vendor}:{product}:{version}:{update}:{edition}:{language}`。
- 从漏洞描述中提取关键信息（如产品名称、版本、影响范围等），验证模型输出是否与描述匹配。
- 如果漏洞描述缺乏足够的信息，保持字段为空或省略，不得添加或虚构信息。

### 输出格式：
请以 JSON 对象格式输出，包含以下三个字段：
- `flag`: 布尔值。`true` 表示模型输出正确，`false` 表示不正确。
- `detail`: 详细解释判断依据，包括输出是否准确的原因、是否符合描述，以及正确答案的推导过程。
- `cpe`: 一个数组，包含从漏洞描述中提取或生成的正确 CPE 字符串。

### 输入信息：
#### 原始漏洞情报：
{vulnerability_intelligence}

#### 大模型输出：
{ai_output}

请根据以上信息进行详细分析并输出结果。

```

### 示例替换
将 `{vulnerability_intelligence}` 替换为原始漏洞情报：
```plaintext
{'vuln_name': 'JerryScript存在未明漏洞CNVD-2022-07242', 'vuln_desc': 'JerryScript是JerryScriptJerryscript项目的一款轻量级的JavaScript引擎JerryScript 3.0.0 存在安全漏洞该漏洞源于 /parser/js/js-scanner.c(scanner_scan_statement_end) 有一个断言 context_p-stack_top_uint8 == SCAN_STACK_TRY_STATEMENT || context_p-stack_top_uint8 == SCAN_STACK_CATCH_STATEMENT 失败 目前没有详细的漏洞细节提供', 'effect_scope': None}
```

将 `{ai_output}` 替换为模型生成的输出：
```plaintext
{
  "cpe": [
    "cpe:/a:jerryscript:jerryscript:3.0.0"
  ]
}
```

### 输出要求
根据判断，输出的 JSON 格式结果如下：
```json
{
  "flag": true,
  "detail": "大模型的输出是正确的。漏洞描述中明确提到 JerryScript 3.0.0 存在安全漏洞，而输出的CPE值 'cpe:/a:jerryscript:jerryscript:3.0.0' 完全符合CPE标准，且与漏洞描述中的信息一致。",
  "cpe": [
    "cpe:/a:jerryscript:jerryscript:3.0.0"
  ]
}
```

### 任务提示设计（英文版Prompt）

```plaintext
You are a world-class cybersecurity expert specializing in analyzing and processing vulnerability intelligence. Based on the provided vulnerability description and the AI model's output, assess whether the model's output is correct and produce results according to the following requirements.

### Task Requirements:
1. **Evaluate Output Accuracy**: Verify if the AI model's CPE output matches the original vulnerability description.
2. **Generate Correct Answer**: If the model's CPE output is incorrect, generate the correct CPE according to the vulnerability description.

### Evaluation Criteria:
- The AI model's CPE output must follow the exact format: `cpe:/{part}:{vendor}:{product}:{version}:{update}:{edition}:{language}`.
- Extract key details from the vulnerability description (e.g., product name, version, affected scope) and check if the model's output aligns with the description.
- If the description lacks sufficient details to produce a complete CPE, keep the relevant fields empty or omit them. Do not add or fabricate information.

### Output Format:
Provide the result as a JSON object with the following three keys:
- `flag`: A boolean value where `true` indicates the model's output is correct, and `false` indicates it is incorrect.
- `detail`: A detailed explanation of the assessment, including reasons why the output is accurate or inaccurate, and the process of deriving the correct answer.
- `cpe`: An array containing the correct CPE strings extracted or generated from the vulnerability description.

### Input Information:
#### Original Vulnerability Description:
{vulnerability_intelligence}

#### AI Model Output:
{ai_output}

Based on the above information, perform a detailed analysis and provide the result.
```