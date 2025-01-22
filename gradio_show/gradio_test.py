import os
import re
import json
import pandas as pd
import gradio as gr

# 定义路径
input_path = "/root/data/wjy/vip_vul_pro/RAG_VIP_VULN/llm/output_qwen72B_cpe_new_prompt_eng"
result_file = "/root/data/wjy/vip_vul_pro/RAG_VIP_VULN/gradio_show/result_72B_eng.txt"

# 定义提取 CPE 内容的正则表达式
cpe_pattern = r'{cpe:\/[aho]:([^:]*):([^:]*):([^:]*):([^:]*):([^:]*):([^:]*)}'  # 匹配 JSON 格式的 CPE 数据

# 函数：从文件提取 CPE 数据
def extract_cpe_from_file(filepath):
    """
    从单个文件中提取符合 CPE 格式的数据。
    """
    with open(filepath, "r", encoding="utf-8") as file:
        content = file.read()  # 读取文件内容
        matches = re.findall(cpe_pattern, content)  # 使用正则提取 CPE 数据块
        cpe_data = []
        for match in matches:
            try:
                json_data = json.loads(match)  # 解析 JSON 数据
                cpe_data.extend(json_data["cpe"])  # 提取 "cpe" 键的值
            except json.JSONDecodeError:
                print(f"Invalid JSON in file {filepath}: {match}")
                continue
        return cpe_data

# 函数：从目录提取所有 CPE 数据并保存到 result.txt
def extract_and_save_cpe(input_path, result_file):
    """
    从 input_path 提取 CPE 数据，并将其写入 result.txt 文件。
    """
    all_cpe = []
    files = sorted(os.listdir(input_path))  # 按文件名排序，确保顺序一致
    for filename in files:
        if filename.endswith(".txt"):  # 只处理 .txt 文件
            filepath = os.path.join(input_path, filename)
            cpe_list = extract_cpe_from_file(filepath)
            all_cpe.extend(cpe_list)  # 不添加文件编号，只保留 CPE 数据

    # 将提取到的数据写入 result.txt
    with open(result_file, "w", encoding="utf-8") as file:
        file.write("\n".join(all_cpe))
    print(f"Extracted CPE data saved to {result_file}")

# 函数：读取 result.txt 并解析 CPE 数据
def read_and_parse_cpe(file_path):
    """
    读取 result.txt 文件并解析 CPE 数据为 DataFrame。
    """
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist.")
        return pd.DataFrame(columns=["Part", "Vendor", "Product", "Version", "Other"])  # 返回空 DataFrame

    # 读取 result.txt 文件
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    parsed_cpe_data = []
    for line in lines:
        line = line.strip()
        if line:
            try:
                parts = line.split(":")  # 按冒号分割 CPE 字段
                parsed_cpe_data.append({
                    "Part": parts[1] if len(parts) > 1 else None,
                    "Vendor": parts[2] if len(parts) > 2 else None,
                    "Product": parts[3] if len(parts) > 3 else None,
                    "Version": parts[4] if len(parts) > 4 else None,
                    "Other": ":".join(parts[5:]) if len(parts) > 5 else None,
                })
            except ValueError:
                print(f"Invalid line format in result.txt: {line}")
                continue

    # 转换为 Pandas DataFrame
    return pd.DataFrame(parsed_cpe_data)

# 如果 result.txt 不存在，生成该文件
if not os.path.exists(result_file):
    print(f"{result_file} 文件不存在，正在生成...")
    extract_and_save_cpe(input_path, result_file)

# Gradio 功能：从 result.txt 文件加载并展示 CPE 数据
def display_cpe_data():
    """
    从 result.txt 加载并返回解析的 CPE 数据。
    """
    df = read_and_parse_cpe(result_file)
    if df.empty:
        print("No valid CPE data found.")
        return pd.DataFrame(columns=["Part", "Vendor", "Product", "Version", "Other"])
    print(f"Parsed DataFrame:\n{df}")
    return df[["Part", "Vendor", "Product", "Version", "Other"]]  # 去掉 File 列，只返回其他列

# 创建 Gradio 界面
with gr.Blocks() as demo:
    with gr.Row():
        gr.Markdown("### 从 result.txt 文件加载并展示 CPE 数据")
    with gr.Row():
        output_data = gr.Dataframe(
            label="解析的 CPE 数据",
            headers=["Part", "Vendor", "Product", "Version", "Other"],
            interactive=False  # 禁止用户编辑数据
        )
    demo.load(fn=display_cpe_data, outputs=output_data)

# 启动 Gradio 界面
demo.launch(server_name="10.0.81.173", server_port=58085)
