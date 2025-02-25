import os
import torch
import pandas as pd
from pymilvus import MilvusClient
from open_clip import get_tokenizer, create_model_from_pretrained
from PIL import Image
import json
import open_clip

# 连接到 Milvus
client = MilvusClient(uri="http://localhost:19530")

# Load the model and config files from the Hugging Face Hub
model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

print("model:",model) # CustomTextCLIP

print("preprocess:",preprocess)
# Compose(
#     Resize(size=224, interpolation=bicubic, max_size=None, antialias=None)
#     CenterCrop(size=(224, 224))
#     <function _convert_to_rgb at 0x000001D2D70BBBE0>
#     ToTensor()
#     Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
# )

tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

print("tokenizer:",tokenizer)  #tokenizer: <open_clip.tokenizer.HFTokenizer object at 0x0000024F57CEB2B0>

# 创建文本和图像 collection
text_collection_name = "wjy_text_embedding_collection"
if text_collection_name not in client.list_collections():
    client.create_collection(
        collection_name=text_collection_name,
        dimension=512  # BiomedCLIP 输出的文本特征维度
    )

image_collection_name = "wjy_image_embedding_collection"
if image_collection_name not in client.list_collections():
    client.create_collection(
        collection_name=image_collection_name,
        dimension=512  # BiomedCLIP 输出的图像特征维度
    )

# 读取 Excel 文件
excel_file = '../DMSA图像文本描述数据.xlsx'  # 请替换为你实际的文件路径
df = pd.read_excel(excel_file)

# 获取图像描述和图像名称列的数据
descriptions = df['Image_description'].tolist()
image_names = df['Image'].tolist()

# 设置图像文件所在的路径
image_folder_path = '../../datasets/images'  # 改为正确的相对路径

# 设备设置：如果有GPU，则使用GPU
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
model.eval()

# 遍历每一行，生成文本特征和图像特征并插入到 Milvus
for idx, (description, image_name) in enumerate(zip(descriptions, image_names)):

    text_data = []
    image_data = []

    # 只处理 JPG 文件
    if not image_name.lower().endswith('.jpg'):
        continue  # 跳过非 JPG 文件

    # 处理文本
    text_feature = tokenizer([description], context_length=256).to(device)
    text_result = model.encode_text(text_feature)
    # print("=========text_result================",text_result)
    text_vector = text_result[0].detach().cpu().numpy().tolist()
    # 存储文本数据
    text_data.append({"id": idx, "vector": text_vector, "text": description})

    # 处理图像
    try:
        img_path = os.path.join(image_folder_path, image_name)  # 获取图像完整路径
        print(img_path) #../datasets\0210.jpg

        raw_image = Image.open(img_path).convert('RGB')  # 打开图像
        image_feature = preprocess(raw_image).unsqueeze(0).to(device)
        image_result = model.encode_image(image_feature)# 转换为PyTorch tensor并送入GPU
        print("image_result",image_result)

        # 确保在转换之前移除梯度信息
        image_vectors = image_result[0].detach().cpu().numpy().tolist()  # 先断开梯度，再转换为列表

        # 存储图像数据
        image_data.append({"id": idx, "vector": image_vectors, "text": image_name})

    except Exception as e:
        print(f"Error processing image {image_name}: {e}")

    # # 将文本数据插入到 Milvus
    # if text_data:
    #     client.insert(collection_name=text_collection_name, data=text_data)
    #     print(f"Successfully inserted {len(text_data)} text embeddings into Milvus.")

    # 将图像数据插入到 Milvus
    if image_data:
        client.insert(collection_name=image_collection_name, data=image_data)
        print(f"Successfully inserted {len(image_data)} image embeddings into Milvus.")
