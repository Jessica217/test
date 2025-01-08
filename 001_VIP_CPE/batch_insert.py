from pymilvus import connections, MilvusClient
import random
import json
from milvus_model.hybrid import BGEM3EmbeddingFunction

# 初始化客户端连接，指定 Milvus 服务器的 URI 和数据库名称
client = MilvusClient(uri="http://10.0.81.173:19530")

# 创建 Milvus 集合（如果集合不存在）
collection_name = "wjy_new_demo3"
if collection_name not in client.list_collections():
    client.create_collection(
        collection_name=collection_name,
        dimension=1024  # 指定向量维度（需匹配嵌入模型的输出维度）
    )

# 加载嵌入模型
model_name_or_path = '/data/huggingface_models/bge-m3' # 可以变
bge_m3_ef = BGEM3EmbeddingFunction(
    model_name=model_name_or_path,  # 指定模型路径
    device='cuda:0',  # 使用的设备（如 'cuda:0' 表示 GPU）
    use_fp16=False  # 是否使用 fp16 模式（在 CPU 上应设置为 False）
)

# 读取 JSON 文件并加载其内容
file_path = '/data/wjy/vip_vul_pro/RAG_VIP_VULN/llm/cpe_data.json'
with open(file_path, 'r', encoding='utf-8') as file:
    docs = json.load(file)

# 创建一个列表以存储要插入的数据
batch_size = 10000  # 每批次插入的数据量
total_docs = len(docs)  # 总文档数量

# 处理并插入数据的批次， 防止 out of memory!!!!
for start_idx in range(0, total_docs, batch_size):
    end_idx = min(start_idx + batch_size, total_docs)
    print(f"Processing batch from {start_idx} to {end_idx}")

    # 获取当前批次的文档数据
    batch_docs = docs[start_idx:end_idx]
    target_data = []

    # 遍历当前批次的文档
    for index, doc in enumerate(batch_docs, start=start_idx):
        print(f"Processing doc {index}")
        # 将字符串形式的 JSON 对象加载为 Python 字典
        doc_dict = json.loads(doc)

        # 使用嵌入模型对文档内容生成嵌入向量
        doc_embedding = bge_m3_ef.encode_documents([doc])

        # 提取密集向量
        dense_vector = doc_embedding["dense"][0]

        # 将 JSON 字段值作为额外字段放入 data_dict
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

        # 将数据添加到目标列表中
        target_data.append(data_dict)

    # 插入当前批次的数据到 Milvus 集合
    res = client.insert(collection_name=collection_name, data=target_data)
    print(f"Batch {start_idx}-{end_idx} insert result:", res)

print("All data insertion completed.")
