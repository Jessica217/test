from pymilvus import connections, Collection, utility, FieldSchema, CollectionSchema, DataType
import numpy as np
def cpe_to_json(cpe_str):
    # 去掉前缀 "cpe:/"
    parts = cpe_str.replace("cpe:/", "").split(":")

    # CPE字段映射
    fields = ["part", "vendor", "product", "version", "update", "edition", "language"]

    # 将CPE字段与内容对应
    cpe_json = {fields[i]: parts[i] if i < len(parts) else "" for i in range(len(fields))}

    # 返回JSON格式
    return json.dumps(cpe_json, indent=4)
# 初始化客户端连接，指定 Milvus 服务器的 URI
connections.connect(alias="default", host="10.0.81.173", port="19530")

# 创建或获取 Milvus 集合
collection_name = "wjy_new_demo3"
collection = Collection(name=collection_name)

# if not utility.has_collection(collection_name):
#     # 如果集合不存在，定义集合的模式并创建集合
#     fields = [
#         FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
#         FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024)
#     ]
#     schema = CollectionSchema(fields, description="Test collection for embeddings")
#     collection = Collection(name=collection_name, schema=schema)
#     print(f"Collection {collection_name} created.")
# else:
#     collection = Collection(name=collection_name)
#     print(f"Collection {collection_name} loaded.")

# 假设我们已经有了一个需要查询的向量，这个向量从文件中加载
# dense_vector = np.load('/root/data/wjy/vip_vul_pro/RAG_VIP_VULN/llm/single_embedding.npy')
src_cpe='cpe:/a:sourcecodester:try_my_recipe:*:*:*:'
src_cpe_json=''
# 使用搜索方法查找最相似的嵌入向量
search_params = {
    "metric_type": "COSINE",
    "params": {"nprobe": 10}
}
results = collection.search(
    data=[dense_vector.tolist()], anns_field="vector", param=search_params, limit=5, output_fields=["cpe_json"], expr=None
)


cpe:/a:sourcecodester:try_my_recipe:*:*:*:

# 打印出 top 5 的结果和得分
for hits in results:
    for hit in hits:
        print(f"ID: {hit.id}, Score: {hit.score}, CPE JSON: {hit.entity.get('cpe_json')}")

# 断开连接
connections.disconnect(alias="default")
