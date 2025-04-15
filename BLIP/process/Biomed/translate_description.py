from openai import OpenAI
import pandas as pd
import concurrent.futures


client = OpenAI(api_key="sk-feb7addb3f60456fbcfdf5ff7bfbbf49", base_url="https://api.deepseek.com")


def translate_text(text):
    """调用 DeepSeek API 翻译文本"""
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "user", "content": f"请将以下医学描述翻译为英文：{text},请注意，只输出翻译的内容，其余分析不要输出，不要输出。"},
            ],
            stream=False,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"翻译失败: {e}")
        return "翻译失败"


def process_excel(input_file, output_file):
    """读取 Excel，进行多线程翻译，并保存到新文件"""
    # 读取 Excel，假设 'Description' 列包含需要翻译的文本
    df = pd.read_excel(input_file, sheet_name="recheck")

    if "Description" not in df.columns:
        print("错误：Excel 文件中没有 'Description' 列！")
        return

    # 使用多线程加速翻译
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        translated_texts = list(executor.map(translate_text, df["Description"].astype(str)))

    # 将翻译结果写入新列
    df["Translated_Description"] = translated_texts

    # 保存到新的 Excel 文件
    df.to_excel(output_file, index=False)
    print(f"翻译完成，结果已保存至 {output_file}")


# 示例用法
input_file = "output.xlsx"  # 你的输入文件路径
output_file = "translated_recheck.xlsx"  # 你的输出文件路径
process_excel(input_file, output_file)





