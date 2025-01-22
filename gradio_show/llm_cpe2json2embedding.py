from milvus_model.hybrid import BGEM3EmbeddingFunction
import json
import numpy as np
from pymilvus import connections, Collection, utility, FieldSchema, CollectionSchema, DataType

# 加载嵌入模型
model_name_or_path = '/data/huggingface_models/bge-m3'
bge_m3_ef = BGEM3EmbeddingFunction(
    model_name=model_name_or_path,
    device='cuda:1',
    use_fp16=False
)

def cpe_to_json(cpe_str):
    # 去掉前缀 "cpe:/"
    parts = cpe_str.replace("cpe:/", "").split(":")

    # CPE字段映射
    fields = ["part", "vendor", "product", "version", "update", "edition", "language"]

    # 将CPE字段与内容对应
    cpe_json = {fields[i]: parts[i] if i < len(parts) else "" for i in range(len(fields))}

    # 返回JSON格式
    return json.dumps(cpe_json, indent=4)

# 读取 JSON 文件并加载其内容
file_path = '/root/data/wjy/vip_vul_pro/RAG_VIP_VULN/llm/gradio_output_step/first_cpes_converted.json'

with open(file_path, 'r', encoding='utf-8') as file:
    docs = json.load(file)

# 假设我们只处理第一个文档
doc = docs[9] if len(docs) > 0 else None
print(doc)
print(type(doc))

if doc:
    # 使用嵌入模型对单个文档的内容生成嵌入向量

    query = json.dumps(doc, indent=4)
    print(f"query:{query}")
    doc_embedding = bge_m3_ef.encode_queries([query])

    # 提取密集向量
    dense_vector = doc_embedding['dense'][0]
    print(dense_vector)

    # 保存嵌入向量到.npy文件
    np.save('/root/data/wjy/vip_vul_pro/RAG_VIP_VULN/llm/single_embedding.npy', dense_vector)

    # # 如果需要，也可以转换为列表后保存为JSON
    # dense_vector_list = dense_vector.tolist()
    # with open('/root/data/wjy/vip_vul_pro/RAG_VIP_VULN/llm/single_embedding.json', 'w', encoding='utf-8') as f_json:
    #     json.dump(dense_vector_list, f_json, indent=4)

    print("Embedding saved successfully.")

    connections.connect(alias="default", host="10.0.81.173", port="19530")

    # 创建或获取 Milvus 集合
    collection_name = "wjy_new_demo3"
    collection = Collection(name=collection_name)

    # 使用搜索方法查找最相似的嵌入向量
    search_params = {
        "metric_type": "COSINE",
        "params": {
            'nprobe': 20, # 在每一层扩展20个候选节点
            'level': 3,
            'radius': 0.8, # 外边界
            'range_filter': 1 # 内边界  cosine中range_filter最大是1 [-1,1]
        }

    }
    results = collection.search(
        data=[dense_vector], anns_field="vector", param=search_params, limit=5, output_fields=["cpe_json"], expr=None
    )

    # 打印出 top 5 的结果和得分
    for hits in results:
        for hit in hits:
            # 计算分数，将其乘以 100 并格式化为两位小数
            formatted_score = "{:.2f}".format(hit.score * 100)
            print(f"ID: {hit.id}, Score: {formatted_score}, CPE JSON: {hit.entity.get('cpe_json')}")

    # 断开连接
    connections.disconnect(alias="default")
else:
    print("No document to process.")


