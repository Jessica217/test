````
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
````

