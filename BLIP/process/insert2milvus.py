import os
import torch
import pandas as pd
from transformers import BlipProcessor, BlipModel, BlipForConditionalGeneration
from pymilvus import connections, MilvusClient
from PIL import Image

# 连接到 Milvus
client = MilvusClient(uri="http://10.0.81.173:19530")

# 加载BLIP模型和处理器
model_path = '/data/wjy/blip-image-captioning-large'  # 你可以根据需要替换成其他预训练模型
processor = BlipProcessor.from_pretrained(model_path)
text_model = BlipModel.from_pretrained(model_path).to("cuda")  # 使用GPU加速
image_model = BlipForConditionalGeneration.from_pretrained(model_path).to("cuda")

# 加载 Excel 文件
excel_file = 'DMSA图像文本描述数据.xlsx'  # 请替换为你实际的文件路径
df = pd.read_excel(excel_file)

# 获取 "Image_description" 和 "Image" 列的数据
descriptions = df['Image_description'].tolist()
image_names = df['Image'].tolist()

# 创建文本 collection
text_collection_name = "wjy_text_embedding_collection"
if text_collection_name not in client.list_collections():
    client.create_collection(
        collection_name=text_collection_name,
        dimension=512  # BLIP 输出的特征维度
    )

# 创建图像 collection
image_collection_name = "high_dim_wjy_image_embedding_collection"
if image_collection_name not in client.list_collections():
    client.create_collection(
        collection_name=image_collection_name,
        dimension=1024
    )


# 设置图像文件所在的路径
image_folder_path = '../datasets/images'  # 这里需要替换为图像所在目录的路径

# 遍历每一行，生成文本特征和图像特征并插入到 Milvus
for idx, (description, image_name) in enumerate(zip(descriptions, image_names)):

    text_data = []
    image_data = []
    # 只处理 JPG 文件
    if not image_name.lower().endswith('.jpg'):
        continue  # 跳过非 JPG 文件

    # 处理文本
    text_inputs = processor(text=description, return_tensors="pt").to("cuda")  # 转换为PyTorch tensor并送入GPU
    with torch.no_grad():
        text_outputs = text_model.get_text_features(**text_inputs)  # 只需要文本特征
    text_vector = text_outputs[0].cpu().numpy().tolist()

    # 存储文本数据
    text_data.append({"id": idx, "vector": text_vector, "text": description})

    # 处理图像
    try:
        img_path = os.path.join(image_folder_path, image_name)  # 获取图像完整路径
        raw_image = Image.open(img_path).convert('RGB')  # 打开图像
        image_inputs = processor(raw_image, return_tensors="pt").to("cuda")  # 转换为PyTorch tensor并送入GPU
        with torch.no_grad():
            out = image_model.vision_model(**image_inputs).last_hidden_state
            # out = image_model.vision_model(**image_inputs).pooler_output
        image_vectors = out[:, 0, :].detach().cpu().numpy().tolist() # pooler_output
        image_vectors = image_vectors[0]

        # 存储图像数据
        image_data.append({"id": idx, "vector": image_vectors, "text": image_name})

    except Exception as e:
        print(f"Error processing image {image_name}: {e}")

    # # 将文本数据插入到 Milvus
    # client.insert(collection_name=text_collection_name, data=text_data)
    # print(f"Successfully inserted {len(text_data)} text embeddings into Milvus.")

    #将图像数据插入到 Milvus
    client.insert(collection_name=image_collection_name, data=image_data)
    print(f"Successfully inserted {len(image_data)} image embeddings into Milvus.")

