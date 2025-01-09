from clickhouse_driver import Client
import json
import re

# 设置ClickHouse数据库连接信息
host = '10.0.81.173'
user = 'root'
password = 'white_hat@202406'
database = 'vip_vul'
table = 'vip_cpe'

def get_database_connection():
    """获取ClickHouse数据库连接"""
    client = Client(
        host=host,
        user=user,
        password=password,
        database=database
    )
    return client

def cpe_to_json(cpe_str):
    # 去掉前缀 "cpe:/"
    parts = cpe_str.replace("cpe:/", "").split(":")

    # CPE字段映射
    fields = ["part", "vendor", "product", "version", "update", "edition", "language"]

    # 将CPE字段与内容对应
    cpe_json = {fields[i]: parts[i] if i < len(parts) else "" for i in range(len(fields))}

    # 返回JSON格式
    return json.dumps(cpe_json, indent=4)

def fetch_and_format_cpe_data():
    """从ClickHouse中获取cpe_item_name字段并格式化为JSON"""
    client = get_database_connection()
    query = f"SELECT cpe_item_name FROM {table}"
    result = client.execute(query)
    # print(result)
    # result = 'cpe:/a:wenlin_institute:wenlin:1.0'
    formatted_cpe_data = [cpe_to_json(row[0]) for row in result if row[0] is not None]

    with open("cpe_data.json", "w+", encoding="utf-8") as f:
        json.dump(formatted_cpe_data, f, indent=4, ensure_ascii=False, separators=(',', ': '))

if __name__ == "__main__":
    fetch_and_format_cpe_data()
