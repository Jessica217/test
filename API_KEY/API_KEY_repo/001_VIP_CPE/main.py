# 连接到clickhouse数据库，然后cpe2json，将json保存到cpe_data.json中；
# 然后使用batch_insert.py脚本连接mivlus 数据库，创建新集合，wjy_demo4, embedding_model = bge_m3,然后向量化到向量数据库中。

#001_API
from fastapi import FastAPI, HTTPException
import json
import uvicorn
from clickhouse_driver import Client
from pymilvus import connections, MilvusClient
from milvus_model.hybrid import BGEM3EmbeddingFunction
import configparser


# insert新数据 如果是爬虫或者新的数据，先把他导入到向量数据库中，然后再进行匹配
# 输入是vip_cpe（那么格式已经是固定的），然后存入到向量数据库中  数据格式：cpe:/a:wenlin_institute:wenlin:1.0

#
# 读取配置文件
config = configparser.ConfigParser()
config.read('config.ini')
app = FastAPI()

def get_database_connection():
    """获取ClickHouse数据库连接"""
    client = Client(
        host=config['CLICKHOUSE']['Host'],
        user=config['CLICKHOUSE']['User'],
        password=config['CLICKHOUSE']['Password'],
        database=config['CLICKHOUSE']['Database']
    )
    return client

def fetch_and_format_cpe_data(table):
    """从ClickHouse中获取cpe_item_name字段的前100条记录并格式化为JSON"""
    client = get_database_connection()
    # 添加LIMIT 10到你的查询以只获取前10条记录
    query = f"SELECT cpe_item_name FROM {table} LIMIT 100"
    result = client.execute(query)
    formatted_cpe_data = [cpe_to_json(row[0]) for row in result if row[0] is not None]
    # 将每个结果行格式化为JSON
    formatted_cpe_data = [cpe_to_json(row[0]) for row in result if row[0] is not None]
    return formatted_cpe_data



def cpe_to_json(cpe_str):
    parts = cpe_str.replace("cpe:/", "").split(":")
    fields = ["part", "vendor", "product", "version", "update", "edition", "language"]
    cpe_json = {fields[i]: parts[i] if i < len(parts) else "" for i in range(len(fields))}
    return json.dumps(cpe_json, indent=4)


@app.post("/input_cpe_list")
async def process_data():
    # Fetch data and process
    docs = fetch_and_format_cpe_data(config['CLICKHOUSE']['Table'])
    print(type(docs))

    # Connect to Milvus
    connections.connect("default", host=config['MILVUS']['Host'], port=config['MILVUS']['Port'])
    client = MilvusClient(connection="default")
    collection_name = config['MILVUS']['CollectionName']

    # Create collection if not exists
    if collection_name not in client.list_collections():
        client.create_collection(
            collection_name=collection_name,
            dimension=int(config['MILVUS']['Dimension'])
        )

    # Load embedding model
    bge_m3_ef = BGEM3EmbeddingFunction(
        model_name_or_path=config['MODEL']['ModelPath'],
        device=config['MODEL']['Device'],
        use_fp16=config['MODEL']['UseFP16'].lower() == 'true'
    )

    batch_size = 10 # Batch size for processing to avoid memory issues
    total_docs = len(docs)
    for start_idx in range(0, total_docs, batch_size):
        end_idx = min(start_idx + batch_size, total_docs)
        print(f"Processing batch from {start_idx} to {end_idx}")
        batch_docs = docs[start_idx:end_idx]
        target_data = []
        for index, doc in enumerate(batch_docs, start=start_idx):
            doc_dict = json.loads(doc)
            doc_embedding = bge_m3_ef.encode_documents([doc])
            dense_vector = doc_embedding["dense"][0]
            data_dict = {
                "id": index,
                "vector": dense_vector,
                "cpe_json": doc,
                "part": doc_dict.get("part", ""),  # 获取 'part' 字段
                "vendor": doc_dict.get("vendor", ""),  # 获取 'vendor' 字段
                "product": doc_dict.get("product", ""),  # 获取 'product' 字段
                "version": doc_dict.get("version", ""),  # 获取 'version' 字段
                "update": doc_dict.get("update", ""),  # 获取 'update' 字段
                "edition": doc_dict.get("edition", ""),  # 获取 'edition' 字段
                "language": doc_dict.get("language", "")  # 获取 'language' 字段
            }
            target_data.append(data_dict)

        # Insert batch data to Milvus
        res = client.insert(collection_name=collection_name, data=target_data)
        print(f"Batch {start_idx}-{end_idx} insert result:", res)

    return {"status": "200 OK", "message": "Data processed successfully"}


# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8009)

