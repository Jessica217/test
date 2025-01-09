from clickhouse_driver import Client
import os
import pandas as pd
import re

# ClickHouse配置信息
host = '10.0.81.173'
user = 'root'
password = 'white_hat@202406'
database = 'simple_vip_vul'
table = 'vip_vuln'

# 定义CSV文件路径
csv_file_path = '/data/xw/dataset/vip_vuln.csv'
cleaned_csv_file_path = '/data/wjy/vip_vul_pro/RAG_VIP_VULN/vip_vuln_cleaned.csv'


# Step 1: 读取并清理CSV文件
def read_and_clean_csv(csv_file_path, cleaned_file_path):
    # 仅选择指定的列
    columns_to_keep = ['vuln_name', 'vuln_desc', 'effect_scope', 'safe_version', 'vuln_reappear']
    data = pd.read_csv(csv_file_path, usecols=columns_to_keep)

    # 清理文本中的引号和特殊字符
    def clean_text(text):
        if pd.isnull(text):
            return ''
        text = str(text).strip().replace('"', '').replace("'", "")
        text = re.sub(r'[^\x20-\x7E\u4e00-\u9fa5]+', '', text)
        return text

    # 应用清理函数到所有列
    data = data.applymap(clean_text)
    data = data.fillna('')

    # 保存清理后的CSV文件
    data.to_csv(cleaned_file_path, index=False, encoding='utf-8')
    print("清理后的CSV文件已保存。")
    return cleaned_file_path


# Step 2: 连接ClickHouse，创建数据库和表
def create_database_and_table(client):
    # 创建数据库
    client.execute(f"CREATE DATABASE IF NOT EXISTS {database}")

    # 删除旧表格（如果存在）
    client.execute(f"DROP TABLE IF EXISTS {database}.{table}")
    print("旧表格已删除。")

    # 创建表，定义字段
    create_table_query = f"""
    CREATE TABLE IF NOT EXISTS {database}.{table} (
        id UInt64,
        vuln_name String,
        vuln_desc Nullable(String),
        effect_scope Nullable(String),
        vuln_reappear Nullable(String),
        safe_version Nullable(String)
    ) ENGINE = MergeTree()
    ORDER BY id;
    """
    client.execute(create_table_query)
    print("数据库和表创建完成。")


# Step 3: 使用cat命令将清理后的CSV数据导入ClickHouse表
def insert_data_to_clickhouse(csv_file, host, user, password, database, table):
    insert_data_command = f"cat {csv_file} | sudo docker exec -i clickhouse-server clickhouse-client --host {host} --user {user} --password {password} --query=\"INSERT INTO {database}.{table} FORMAT CSVWithNames\""
    os.system(insert_data_command)
    print("数据已成功导入ClickHouse表中。")


# 主执行流程
if __name__ == "__main__":
    # 清理CSV文件
    cleaned_csv_file_path = read_and_clean_csv(csv_file_path, cleaned_csv_file_path)

    # 初始化ClickHouse客户端
    client = Client(host=host, user=user, password=password)

    # 创建数据库和表
    create_database_and_table(client)

    # 将清理后的CSV数据导入ClickHouse表
    insert_data_to_clickhouse(cleaned_csv_file_path, host, user, password, database, table)
