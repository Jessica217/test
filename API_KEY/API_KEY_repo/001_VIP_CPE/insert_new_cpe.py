from fastapi import FastAPI, HTTPException
import json
import uvicorn
from clickhouse_driver import Client
from pymilvus import connections, MilvusClient, CollectionSchema, FieldSchema, DataType, Collection
from milvus_model.hybrid import BGEM3EmbeddingFunction
import configparser
import ast


# 配置文件和 FastAPI 应用初始化
config = configparser.ConfigParser()
config.read('config.ini')
app = FastAPI()


def cpe_2_json(cpe_str):
    parts = cpe_str.replace("cpe:/", "").split(":")
    fields = ["part", "vendor", "product", "version", "update", "edition", "language"]
    cpe_json = {fields[i]: parts[i] if i < len(parts) else "*" for i in range(len(fields))}
    # return json.dumps(cpe_json, indent=4)
    return cpe_json

# # 连接到 Milvus
# connections.connect("default", host=config['MILVUS']['Host'], port=config['MILVUS']['Port'])
# client = MilvusClient(connection="default")
# res = client.query(
#     collection_name="wjy_new_demo3",
#     output_fields=["count(*)"]
# )
@app.post("/process_cpe")
async def vectorize_cpe(data: list[str]): # 输入的data是str类型的，cpe:/h:tophant:tophant:2.19
    results = []
    # 连接到 Milvus
    connections.connect("default", host=config['MILVUS']['Host'], port=config['MILVUS']['Port'])
    client = MilvusClient(connection="default")
    res = client.query(
        collection_name="wjy_new_demo3",
        output_fields=["count(*)"]
    )
    cpe_count = res[0]['count(*)']  # 访问字典中的值
    for cpe_data in data:
        cpe_count += 1
        cpe_json = cpe_2_json(cpe_data)
        print("Count:", cpe_count)
        collection_name = config['MILVUS']['CollectionName']

        # 若集合不存在，则创建集合
        if collection_name not in client.list_collections():
            client.create_collection(
                collection_name=collection_name,
                dimension=int(config['MILVUS']['Dimension'])
            )

        # 加载向量嵌入模型
        bge_m3_ef = BGEM3EmbeddingFunction(
            model_name_or_path=config['MODEL']['ModelPath'],
            device=config['MODEL']['Device'],
            use_fp16=config['MODEL']['UseFP16'].lower() == 'true'
        )

        doc_embedding = bge_m3_ef.encode_documents([json.dumps(cpe_json)])
        dense_vector = doc_embedding["dense"][0]
        data_dict = {
            "id": cpe_count,
            "vector": dense_vector,
            "cpe_json": cpe_json,
            "part": cpe_json.get("part", ""),  # 获取 'part' 字段
            "vendor": cpe_json.get("vendor", ""),  # 获取 'vendor' 字段
            "product": cpe_json.get("product", ""),  # 获取 'product' 字段
            "version": cpe_json.get("version", ""),  # 获取 'version' 字段
            "update": cpe_json.get("update", ""),  # 获取 'update' 字段
            "edition": cpe_json.get("edition", ""),  # 获取 'edition' 字段
            "language": cpe_json.get("language", "")  # 获取 'language' 字段
        }
        target_data = []
        target_data.append(data_dict)
        print(target_data)
        # 将数据插入到 Milvus
        res = client.insert(collection_name=collection_name, data=target_data)
        results.append(res)
        print(f"Insert result for {cpe_data}: {res}")

        # print(f"Insert result: {res}")

    return {"status": "OK", "message": "CPE data processed and vectorized successfully"}
