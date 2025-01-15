import pandas as pd


# 读取Excel文件，确保第一行作为列名
def read_excel(file_path):
    return pd.read_excel(file_path, header=0)  # 默认header=0，确保第一行作为列名


# 任务 001: 修改第一列“Patients”的值
def modify_patients_column(df):
    # 确保 "Patients" 列名存在
    if "Patients" in df.columns:
        df['Patients'] = df['Patients'].apply(lambda x: f"{int(x):04d}.jpg")
    return df


# 任务 002: 合并H列与J列的内容
def merge_columns_H_J(df):
    # 确保 H 列和 J 列存在
    if "左侧APN" in df.columns and "右侧APN" in df.columns:
        df['H_J_merged'] = df.apply(lambda row: f"左侧肾脏{row['左侧APN']}, 右侧肾脏{row['右侧APN']}", axis=1)
    return df


# 任务 003: 根据I列与K列的值生成描述
def generate_description(df):
    def generate_row_description(row):
        left_kidney = f"左侧肾脏{row['左侧APN']}"
        right_kidney = f"右侧肾脏{row['右侧APN']}"

        # 判断I列和K列的状态，生成描述
        left_status = "属于异常肾脏" if row['左侧output'] == 1 else ""
        right_status = "属于异常肾脏" if row['右侧output'] == 1 else ""

        description = f"{left_kidney}{', ' + left_status if left_status else ''}，" \
                      f"{right_kidney}{', ' + right_status if right_status else ''}，"

        # 判断是否患有APN急性肾盂肾炎
        if row['左侧output'] == 1 or row['右侧output'] == 1:
            description += "综合考虑，该患者患有APN急性肾盂肾炎。"
        else:
            description += "该患者未患APN急性肾盂肾炎。"

        return description

    df['Image_description'] = df.apply(generate_row_description, axis=1)
    return df


# 任务 004: 根据I列和J列的值进行条件判断，得出结论
def generate_diagnosis(df):
    def get_diagnosis(row):
        if row['左侧output'] == 0 and row['右侧output'] == 0:
            return "没有APN急性肾盂肾炎"
        else:
            return "患有APN急性肾盂肾炎"

    df['Diagnosis'] = df.apply(get_diagnosis, axis=1)
    return df


# 任务 005: 生成新的Excel文件
def save_to_excel(df, output_path):
    # 保存结果到新的Excel文件，确保列名正确
    df[['Patients', 'Image_description']].to_excel(output_path, index=False)


# 主函数
def process_excel(file_path, output_path):
    # 读取Excel文件
    df = read_excel(file_path)

    # 执行任务 001 到 005
    df = modify_patients_column(df)
    df = merge_columns_H_J(df)
    df = generate_description(df)
    df = generate_diagnosis(df)

    # 保存结果到新的Excel文件
    save_to_excel(df, output_path)
    print(f"处理完成，结果已保存到 {output_path}")

# 示例调用
file_path = '副本DMSA 损伤数据（简化）.xlsx'  # 输入Excel文件路径
output_path = 'DMSA图像文本描述数据.xlsx'  # 输出Excel文件路径
process_excel(file_path, output_path)
