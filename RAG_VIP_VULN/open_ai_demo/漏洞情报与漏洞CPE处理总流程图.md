

### 一、总流程结构

![漏洞情报提取三要素](D:\a斗象科技\漏洞大模型文档\漏洞情报提取三要素.png)

整个流程可以分为三个主要步骤：**数据存储**、**大模型提取与格式化**、**查询与匹配**。具体步骤如下：

---

### 1. 数据存储流程

   **步骤概述**：将CPE数据从MySQL数据库中提取，转化为适合向量化的格式，存入向量数据库，以便后续的相似性检索。

   **流程详细**：

   1.1. **数据提取**：从MySQL数据库表`vip_cpe`中提取字段`cpe_item_name`（如`"cpe:/a:jenkins:fortify_on_demand:1.0::~~~jenkins~~."`）。

   1.2. **JSON格式转换**：使用`cpe2json`方法将`cpe_item_name`字段转化为JSON格式的字符串，确保结构化的字段表示。

   ```json
   {
       "part": "a",
       "vendor": "jenkins",
       "product": "fortify_on_demand",
       "version": "1.0",
       "update": "",
       "edition": "~~~jenkins~~.",
       "language": ""
   }
   ```

   1.3. **向量化处理**：将JSON格式的CPE数据转换为适合向量化的文本格式，并输入Embedding模型生成向量表示。

   1.4. **向量存储**：将生成的向量及原始数据存入向量数据库，以便后续相似性检索。

   **流程图**：

   ```mermaid
   flowchart TD
       A[从 MySQL 数据库提取 cpe_item_name] --> B[cpe_item_name 转为 JSON 格式]
       B --> C[整理为向量化的文本格式]
       C --> D[使用 Embedding 模型生成向量]
       D --> E[存入向量数据库]
   ```

---

### 2. 大模型CPE提取与格式化流程

   **步骤概述**：使用大模型从漏洞情报文本中自动提取符合CPE格式的字符串，并将其格式化为JSON对象，便于后续查询。

   **流程详细**：

   2.1. **漏洞情报收集**：从外部情报源或内部系统中获取漏洞描述文本。

   2.2. **大模型解析**：使用大模型解析漏洞情报文本，识别并提取所有符合CPE格式的字符串。

   2.3. **格式校验**：判断提取的内容是否符合`cpe:/{part}:{vendor}:{product}:{version}:{update}:{edition}:{language}`标准格式。

   2.4. **JSON格式化输出**：将符合CPE格式的字符串转换为JSON对象，并以`cpe`键为单一键输出JSON结果。

   **提取要求**：
   - **输出格式**：结果为JSON对象，包含单一键`cpe`，其值为包含提取CPE字符串的数组。
   - **字段严格性**：若字段不完整，则保持原样，不进行补充或修改。

   **流程图**：

   ```mermaid
   flowchart TD
       A[获取漏洞情报信息] --> B[解析漏洞情报文本]
       B --> C[使用大模型提取 CPE 格式字符串]
       C --> D{是否符合 CPE 格式？}
       D -- 是 --> E[输出符合 CPE 格式的字符串]
       D -- 否 --> F[丢弃不符合格式的内容]
       E --> G[将 CPE 格式化为 JSON 对象]
       G --> H[输出最终 JSON 结果]
   ```

---

### 3. 查询与匹配流程

   **步骤概述**：在接收到输入的CPE字符串后，转化为向量表示，并在向量数据库中进行相似性检索，以返回最匹配的CPE项。

   **流程详细**：

   3.1. **接收输入CPE**：接收由大模型从漏洞情报中提取的CPE字符串或其他来源的CPE。

   3.2. **JSON格式转换**：使用`cpe2json`方法将输入的CPE字符串转换为JSON格式。

   3.3. **向量生成**：将JSON字符串输入Embedding模型，生成该CPE的向量表示。

   3.4. **相似性检索**：在向量数据库中基于生成的向量进行相似性检索，找到最匹配的CPE项。

   3.5. **结果排序和返回**：将检索结果按相似度排序，输出前几个最匹配的CPE项，并附上相似度得分。

   **流程图**：

   ```mermaid
   flowchart TD
       subgraph 查询与匹配
           F[接收输入的 CPE] --> G[输入 CPE 转为 JSON 格式]
           G --> H[输入 JSON 到 embedding 模型生成向量]
           H --> I[在向量数据库中执行相似性检索]
           I --> J[返回排序的匹配结果]
       end
   ```

---

**综合流程总结**

1. **数据存储阶段**：从数据库中提取并转化CPE数据，将其向量化后存入向量数据库。
2. **大模型提取与格式化阶段**：从漏洞情报文本中提取符合CPE格式的信息，格式化为标准的JSON输出。
3. **查询与匹配阶段**：接收CPE输入后生成向量，在向量数据库中进行相似性检索，返回最匹配的结果。

该流程可以有效支持基于CPE的漏洞情报提取和相似性检索，为安全团队提供精确的匹配结果和数据支持。

### 二、附件

#### 英文版本提示词：

```apl
You are a world-class cybersecurity expert specializing in the analysis and handling of vulnerability intelligence. From the provided vulnerability description, please extract all relevant network security CPEs (Common Platform Enumeration) that adhere to the CPE format. 

**Requirements for Extraction:** 

1. The output should consist only of strings in the exact CPE format: `cpe:/{part}:{vendor}:{product}:{version}:{update}:{edition}:{language}`.
2. If certain fields are missing in the original CPE data, retain the details as they are without adding or modifying any information.

**Output Specifications:** 

- Format the results as a JSON object with a single key, `cpe`, whose value is an array listing each extracted CPE string individually.
- No extra explanations or descriptions are required.

### Vulnerability Intelligence:
```

---

#### 中文版本提示词：

```apl
你是一位全球顶尖的网络安全专家，专注于漏洞情报的分析和处理。请从提供的漏洞描述中提取所有符合CPE格式的网络安全CPE（通用平台枚举）。

**提取要求：**

1. 输出内容仅包含符合CPE标准的字符串，格式为：`cpe:/{part}:{vendor}:{product}:{version}:{update}:{edition}:{language}`。
2. 若某些CPE字段在漏洞情报中缺失，请严格遵循原始信息，不补充或修改任何内容。

**输出格式：**

- 将结果以JSON格式输出，显示为一个包含单一键`cpe`的JSON对象，其值为一个数组，数组中逐一列出每个提取的CPE字符串。
- 不需要添加任何额外的解释或说明。

### 漏洞情报：
```

### 优化后的 Prompt（中文版）

你是一位全球顶尖的网络安全专家，专注于漏洞情报的分析和处理。请从以下漏洞描述中提取所有符合 **CPE 标准** 的网络安全 CPE（通用平台枚举）。提取规则和输出格式如下：

---

### 提取规则

1. **严格遵循 CPE 标准格式：**  
   CPE 字符串必须严格符合以下格式：  cpe:/{part}:{vendor}:{product}:{version}:{update}:{edition}:{language}， 共计 7 个字段。若某字段缺失，必须用占位符 `*` 补充，确保每个字段完整。
1. **严格匹配要求：**  
   - 仅提取完全符合上述标准的 CPE 字符串。
   - 不对原始数据内容进行任何推测或补全，仅基于漏洞描述中的内容。

### 输出要求

1. **输出为 JSON 格式：**  
   返回一个 JSON 对象，包含单一键名 `cpe`，其值为一个数组，数组中列出所有提取到的 CPE 字符串。

2. **格式一致性：**  
   - 数组内字符串按提取顺序排列。
   - 确保每个字段完整，占位符无遗漏或多余。

---

### 示例参考

#### 输入示例 1：

```json
{
  "vuln_name": "Linux kernel 安全漏洞",
  "vuln_desc": "Linux kernel 是美国 Linux 基金会的开源操作系统内核。Linux kernel 存在安全漏洞，攻击者可通过 Frozen EBPF Map Race 提升权限。",
  "effect_scope": null
}
```

#### 输出示例 1：

```json
{
  "cpe": [
    "cpe:/o:linux:linux_kernel:*:*:*:*"
  ]
}
```

#### 输入示例 2：

```json
{
  "vuln_name": "工业控制软件漏洞",
  "vuln_desc": "Mitsubishi Electric 的 MC Works64 和 Iconics Genesis64 软件均存在权限提升漏洞。",
  "effect_scope": "工业控制软件"
}
```

#### 输出示例 2：

```json
{
  "cpe": [
    "cpe:/a:mitsubishi_electric:mc_works64:*:*:*:*",
    "cpe:/a:iconics:genesis64:*:*:*:*"
  ]
}
```

---

### 提取任务

请基于上述规则提取漏洞描述中的 CPE 字符串，确保格式统一、字段完整，并返回符合标准的 JSON 结果。

### 漏洞情报：

-----

### 优化后的 Prompt（英文版）

---

You are a world-class cybersecurity expert specializing in the analysis and handling of vulnerability intelligence. Please extract all cybersecurity CPEs (Common Platform Enumerations) that conform to the **CPE standard** from the following vulnerability descriptions. The extraction rules and output format are as follows:

---

### Extraction Rules

1. **Strictly Follow the CPE Standard Format:**  
   CPE strings must strictly conform to the following format:  
   `cpe:/{part}:{vendor}:{product}:{version}:{update}:{edition}:{language}`, consisting of 7 fields. If any field is missing, it must be filled with the placeholder `*`, ensuring every field is complete.

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
{
  "vuln_name": "Linux kernel security vulnerability",
  "vuln_desc": "The Linux kernel is an open-source operating system kernel developed by the Linux Foundation in the United States. It has a security vulnerability that allows attackers to escalate privileges via the Frozen EBPF Map Race.",
  "effect_scope": null
}
```

#### Output Example 1:

```json
{
  "cpe": [
    "cpe:/o:linux:linux_kernel:*:*:*:*"
  ]
}
```

#### Input Example 2:

```json
{
  "vuln_name": "Industrial control software vulnerability",
  "vuln_desc": "Privilege escalation vulnerabilities exist in Mitsubishi Electric's MC Works64 and Iconics Genesis64 software.",
  "effect_scope": "Industrial control software"
}
```

#### Output Example 2:

```json
{
  "cpe": [
    "cpe:/a:mitsubishi_electric:mc_works64:*:*:*:*",
    "cpe:/a:iconics:genesis64:*:*:*:*"
  ]
}
```

---

### Extraction Task

Based on the rules above, extract all CPE strings from the vulnerability description, ensure the format is consistent, and fields are complete. Return the result in JSON format as specified.

---



以下是包含 `detail` 键的更新版完整 Prompt，明确了规则和输出格式要求：

### 包含detail的 Prompt

你是一位全球顶尖的网络安全专家，专注于漏洞情报的分析和处理。请从以下漏洞描述中提取所有符合 **CPE 标准** 的网络安全 CPE（通用平台枚举）。提取规则和输出格式如下：

---

### 提取规则

1. **严格遵循 CPE 标准格式：**  
   CPE 字符串必须严格符合以下格式：  cpe:/{part}:{vendor}:{product}:{version}:{update}:{edition}:{language}，共计 7 个字段。若某字段缺失，必须用占位符 `*` 补充，确保每个字段完整。

2. **严格匹配要求：**  
   - 仅提取完全符合上述标准的 CPE 字符串。
   - 不对原始数据内容进行任何推测或补全，仅基于漏洞描述中的内容。

3. **字段来源说明：**  
   每个提取出的 CPE 字符串必须提供解释，说明字段的来源。例如：  
   - 若字段如 `vendor` 和 `product` 来源于漏洞描述中的具体内容，需明确指出内容的提取逻辑。  
   - 若字段如 `version` 缺失，用占位符 `*` 填充，同时在解释中注明该字段缺失。  

---

### 输出要求

1. **输出为 JSON 格式：**  
   返回一个 JSON 对象，包含以下两部分：  
   - `cpe`：一个数组，列出所有提取到的 CPE 字符串。  
   - `detail`：一个数组，与 `cpe` 一一对应，解释每个 CPE 的提取依据。

2. **格式一致性：**  
   - 数组内字符串按提取顺序排列。
   - 确保每个字段完整，占位符无遗漏或多余。
   - `detail` 中的解释需清晰简洁，指明每个字段的来源。

---

### 示例参考

#### 输入示例 1：

```json
{
  "vuln_name": "Linux kernel 安全漏洞",
  "vuln_desc": "Linux kernel 是美国 Linux 基金会的开源操作系统内核。Linux kernel 存在安全漏洞，攻击者可通过 Frozen EBPF Map Race 提升权限。",
  "effect_scope": null
}
```

#### 输出示例 1：

```json
{
  "cpe": [
    "cpe:/o:linux:linux_kernel:*:*:*:*"
  ],
  "detail": [
    {
      "cpe": "cpe:/o:linux:linux_kernel:*:*:*:*",
      "explanation": "从漏洞描述中提取出 'Linux kernel' 作为 product，'Linux' 作为 vendor，未提及版本、更新、edition 和语言，因此这些字段用占位符 * 填充。"
    }
  ]
}
```

#### 输入示例 2：

```json
{
  "vuln_name": "工业控制软件漏洞",
  "vuln_desc": "Mitsubishi Electric 的 MC Works64 和 Iconics Genesis64 软件均存在权限提升漏洞。",
  "effect_scope": "工业控制软件"
}
```

#### 输出示例 2：

```json
{
  "cpe": [
    "cpe:/a:mitsubishi_electric:mc_works64:*:*:*:*",
    "cpe:/a:iconics:genesis64:*:*:*:*"
  ],
  "detail": [
    {
      "cpe": "cpe:/a:mitsubishi_electric:mc_works64:*:*:*:*",
      "explanation": "从漏洞描述中提取出 'Mitsubishi Electric' 作为 vendor，'MC Works64' 作为 product，未提及版本、更新、edition 和语言，因此这些字段用占位符 * 填充。"
    },
    {
      "cpe": "cpe:/a:iconics:genesis64:*:*:*:*",
      "explanation": "从漏洞描述中提取出 'Iconics' 作为 vendor，'Genesis64' 作为 product，未提及版本、更新、edition 和语言，因此这些字段用占位符 * 填充。"
    }
  ]
}
```

---

### 提取任务

请基于上述规则提取漏洞描述中的 CPE 字符串，并为每个结果提供解释，确保格式统一、字段完整，并返回符合标准的 JSON 结果。

### 漏洞情报：
----

### 包含detail的 Prompt（English version ）

You are a world-class cybersecurity expert specializing in the analysis and processing of vulnerability intelligence. Please extract all valid **CPE (Common Platform Enumeration)** strings from the following vulnerability description. Extraction rules and output format are as follows:

---

### Extraction Rules

1. **Strict Adherence to the CPE Standard Format:**  
   CPE strings must strictly follow the format:  cpe:/{part}:{vendor}:{product}:{version}:{update}:{edition}:{language}， A total of 7 fields must be included. If any field is missing, it must be replaced with the placeholder `*`, ensuring all fields are complete.

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
{
  "vuln_name": "Linux kernel security vulnerability",
  "vuln_desc": "Linux kernel is the open-source operating system kernel managed by the Linux Foundation. A security vulnerability exists in the Linux kernel, which attackers can exploit via the Frozen EBPF Map Race to escalate privileges.",
  "effect_scope": null
}
```

#### Output Example 1:

```json
{
  "cpe": [
    "cpe:/o:linux:linux_kernel:*:*:*:*"
  ],
  "detail": [
    {
      "cpe": "cpe:/o:linux:linux_kernel:*:*:*:*",
      "explanation": "Extracted 'Linux kernel' as the product and 'Linux' as the vendor from the description. Fields for version, update, edition, and language are not provided, so placeholders '*' are used."
    }
  ]
}
```

#### Input Example 2:

```json
{
  "vuln_name": "Industrial control software vulnerability",
  "vuln_desc": "Mitsubishi Electric's MC Works64 and Iconics' Genesis64 software both have privilege escalation vulnerabilities.",
  "effect_scope": "Industrial control software"
}
```

#### Output Example 2:

```json
{
  "cpe": [
    "cpe:/a:mitsubishi_electric:mc_works64:*:*:*:*",
    "cpe:/a:iconics:genesis64:*:*:*:*"
  ],
  "detail": [
    {
      "cpe": "cpe:/a:mitsubishi_electric:mc_works64:*:*:*:*",
      "explanation": "Extracted 'Mitsubishi Electric' as the vendor and 'MC Works64' as the product from the description. Fields for version, update, edition, and language are not provided, so placeholders '*' are used."
    },
    {
      "cpe": "cpe:/a:iconics:genesis64:*:*:*:*",
      "explanation": "Extracted 'Iconics' as the vendor and 'Genesis64' as the product from the description. Fields for version, update, edition, and language are not provided, so placeholders '*' are used."
    }
  ]
}
```

---

### Extraction Task

Based on the rules above, please extract the CPE strings from the vulnerability description and provide an explanation for each result. Ensure the format is consistent, all fields are complete, and the output conforms to the specified JSON format.

### Vulnerability Description:
