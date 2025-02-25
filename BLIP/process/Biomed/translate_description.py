import openai

# 设置 API 密钥
openai.api_key = 'TIUl43IWuGX4NQWZC6C8D27d1699424bB9D20e00A05fC3F4'

# 设置基础 URL
# openai.base_url = "http://activity.scnet.cn:61080/v1/"
openai.base_url = "https://api.scnet.cn/api/llm/v1/"

# 创建完成任务
completion = openai.chat.completions.create(
    model="DeepSeek-R1-Distill-Qwen-32B",
    messages=[
        {
            "role": "user",
            "content": "How do I output all files in a directory using Python?",
        },
    ],
    stream=False,  # 设置为流式输出
)

print(completion.choices[0].message.content)




