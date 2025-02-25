import os
import torch
import pandas as pd
from pymilvus import MilvusClient, connections, Collection
from open_clip import get_tokenizer, create_model_from_pretrained
from PIL import Image
import json
import numpy as np

# 连接到 Milvus
connections.connect(uri="http://localhost:19530")

model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

image_collection_name = 'wjy_image_embedding_collection'  # 图像向量数据库集合名称
text_collection_name = 'wjy_text_embedding_collection'  # 文本向量数据库集合名称

# 获取集合对象
image_collection = Collection(image_collection_name)
text_collection = Collection(text_collection_name)

# 设备设置：如果有GPU，则使用GPU
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
model.eval()

# 图像编码函数
def encode_image(image_path):
    raw_image = Image.open(image_path).convert('RGB')  # 打开图像
    image_feature = preprocess(raw_image).unsqueeze(0).to(device)
    image_result = model.encode_image(image_feature)  # 转换为PyTorch tensor并送入GPU
    # 确保在转换之前移除梯度信息
    image_vectors = image_result[0].detach().cpu().numpy().tolist()

    return image_vectors

# 图像向量查询函数
def search_image_vectors(query_vector, top_k=20):
    query_vector = np.array([query_vector], dtype=np.float32)
    search_params = {
        "metric_type": "COSINE",
        "params": {'nprobe': 20}  # radius去掉，因为COSINE度量不需要该参数
    }

    # 查询图像向量数据库
    search_results = image_collection.search(
        query_vector,  # 查询向量
        anns_field="vector",  # 向量字段名
        param=search_params,  # 搜索参数
        limit=top_k  # 返回前top_k个结果
    )

    # 解析返回的结果
    if search_results:
        # 仅提取搜索结果中的最相似图像和相似度分数
        matches = []
        for result in search_results[0]:
            matches.append({
                'id': result.id,  # 向量的ID
                'distance': result.distance  # 余弦相似度分数
            })
        return matches
    else:
        print("No results found.")
        return []

# 执行编码和查询
image_vector = encode_image("../extra_50_0003.jpg")
print(image_vector)
image_search_results = search_image_vectors(image_vector, top_k=20)

# 输出查询结果
print("Top 20 search results:")
for match in image_search_results:
    print(f"ID: {match['id']}, Distance: {match['distance']}")
