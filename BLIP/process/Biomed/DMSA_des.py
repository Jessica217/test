import pandas as pd


def generate_description(row):
    left_apn = str(row[0]) if pd.notna(row[0]) else ""
    right_apn = str(row[2]) if pd.notna(row[2]) else ""

    left_status = "属于异常肾脏" if row[1] == 1 else ("" if "正常" in left_apn else "属于正常肾脏")
    right_status = "属于异常肾脏" if row[3] == 1 else ("" if "正常" in right_apn else "属于正常肾脏")

    left_desc = f"左侧肾脏{left_apn}{f'，{left_status}' if left_status else ''}"
    right_desc = f"右侧肾脏{right_apn}{f'，{right_status}' if right_status else ''}"

    if row[1] == 0 and row[3] == 0:
        overall_desc = "综合判断患者未患有APN。"
    else:
        overall_desc = "综合判断患者患有APN。"

    return f"{left_desc}；{right_desc}。{overall_desc}"


def process_excel(file_path, output_path):
    df = pd.read_excel(file_path, usecols=[13, 14, 15, 16])  # 直接按列索引读取 G, H, I, J
    df['Description'] = df.apply(generate_description, axis=1)
    df.to_excel(output_path, index=False)
    print(f"处理完成，结果已保存至 {output_path}")


# 使用示例
input_file = "复查名单.xlsx"  # 替换为你的输入文件路径
output_file = "复查_output.xlsx"  # 替换为你的输出文件路径
process_excel(input_file, output_file)


