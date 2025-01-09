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
openai_api_base = "http://10.0.81.159:8080/v1"

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