import pandas as pd
import json

# 1. 读取 Excel 文件
excel_file = "DMSA图像文本描述数据.xlsx"  # 请替换为你的 Excel 文件路径
output_json_file = "training_data.json"  # 输出 JSON 文件名
image_path_prefix = "BLIP/datasets/images/"  # 图片路径前缀

# 2. 加载 Excel 数据
df = pd.read_excel(excel_file)

# 3. 生成 JSON 数据
data = []
for _, row in df.iterrows():
    image_name = row["Image"]
    image_description = row["Image_description"]
    data.append({
        "image": f"{image_path_prefix}{image_name}",
        "caption": image_description
    })

# 4. 将数据保存为 JSON 文件
with open(output_json_file, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print(f"JSON 文件已成功生成，保存为 {output_json_file}")
