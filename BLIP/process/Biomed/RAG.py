import os
import torch
import pandas as pd
from pymilvus import MilvusClient, connections, Collection
from open_clip import get_tokenizer, create_model_from_pretrained
from PIL import Image
import json
import numpy as np
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer
from llava.model.builder import load_pretrained_model



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

#使用vllm在本地部署llava-med模型
# openai_api_key = "EMPTY"
# openai_api_base = "http://localhost:8000/v1"

llava_tokenizer, llava_model, llava_image_processor, llava_context_len = load_pretrained_model(
        model_path='/work/home/wangjy12023/wjy/LLaVA-Med/llava-med-v1.5-mistral-7b',
        model_base=None,
        model_name='llava-med-v1.5-mistral-7b'
 )

#
# client = OpenAI(
#     api_key=openai_api_key,
#     base_url=openai_api_base,
# )

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
        "params": {'nprobe': 20, 'level': 3, 'radius': 0.8, 'range_filter': 1}  # radius去掉，因为COSINE度量不需要该参数
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


def get_text_from_image_ids(image_ids):
    # 根据图像数据库返回的 ID 查找对应的文本
    results = []
    for image_id in image_ids:
        result = text_collection.query(expr=f"id == {image_id}", output_fields=["text", "id"])  # 确保查询时获取'id'和'text'字段
        if result:  # 确保查询结果非空
            try:
                # 如果返回的数据是字符串类型，转为字典
                if isinstance(result[0], str):
                    data = eval(result[0])  # 将字符串解析为字典
                else:
                    data = result[0]  # 如果是字典则直接使用
                text = data.get('text', None)
                id = data.get('id', None)

                if text:
                    # 确保返回格式一致，text在前，id在后
                    results.append({'text': text, 'id': id})
                else:
                    # 如果没有文本，加入提示信息
                    results.append({'text': 'No text available', 'id': id})
            except Exception as e:
                # 异常处理，打印错误信息
                print(f"Error processing result for image_id {image_id}: {e}")
                results.append({'text': 'Error', 'id': image_id})
        else:
            # 如果查询没有返回结果
            print(f"No result found for image_id {image_id}")
            results.append({'text': 'No text available', 'id': image_id})

    print(f"results: {results}")
    return results


def generate_prompt(query_text, text_search_results):
    # 生成提示词，结合图像和文本查询的结果
    context = " ".join([result['text'] for result in text_search_results])  # 直接访问字典中的 'text' 字段
    prompt = f"""

### Task Description:  
Based on the uploaded imaging data {context}, please answer: {query_text}.  

### Processing Requirements:  

1. **Analysis of Left and Right Kidney Health Conditions**:  
   Based on the SPECT imaging analysis, assess the health status of both kidneys. Please note:  
   - The kidney on the left of the figure is the patient's actual left kidney, and the kidney on the right is the patient's actual right kidney.  
   - Analyze the normal and abnormal conditions of both the left and right kidneys, including radioactive distribution, morphological changes, and other characteristics.  
   - If there is insufficient evidence or the abnormalities cannot be determined from the image, explicitly state "unable to determine" or "unable to draw a conclusion based on the available information."  

2. **Acute Pyelonephritis Diagnosis**:  
   After completing the analysis of kidney health conditions, determine whether the patient has a potential case of **acute pyelonephritis** based on kidney characteristics observed in the imaging. If it cannot be determined or if there is insufficient evidence, explicitly state "unable to determine" or "unable to draw a conclusion based on the available information."  

3. **Output Format Requirements**:  
   - **Assessment of the left kidney** (Normal or abnormal, with specific observations)  
   - **Assessment of the right kidney** (Normal or abnormal, with specific observations)  
   - **Diagnosis of acute pyelonephritis** (Yes / No / Unable to determine)  

4. **Important Notes**:  
   - The analysis must be based solely on the imaging data and the information retrieved from the RAG database. External assumptions or unverified information should not be included.  
   - Clearly describe the conditions of each kidney, avoiding vague or ambiguous statements.  
   - If the available information is insufficient for a definitive conclusion, explicitly state that the determination cannot be made.  

**Final Output Format**:  
- **Left Kidney**: Assessment (Normal or Abnormal, with details)  
- **Right Kidney**: Assessment (Normal or Abnormal, with details)  
- **Acute Pyelonephritis Diagnosis**: (Yes / No / Unable to determine)  
"""
    return prompt

def LLM_generate_answer(client, model_name, user_message):
    completion =  client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_message},
        ],
        temperature=0.5,
        # top_p=0.8,
        top_p=0.6,
        max_tokens=8192,
        extra_body={"repetition_penalty": 1.05}
    )
    return completion.choices[0].message.content

def llava_med_generate_answer(prompt):
    inputs = llava_tokenizer(prompt, return_tensors="pt").to(device)
    output = llava_model.generate(**inputs)
    response = llava_tokenizer.decode(output[0], skip_special_tokens=True)
    return response


# 流程
# def rag_pipeline(image_path, query_text):
#     # 第一步：将图片编码成向量
#     image_vector = encode_image(image_path)
#
#     # 第二步：在图像数据库中检索与图像查询向量最相似的前20个图像向量
#     image_search_results = search_image_vectors(image_vector, top_k=20)
#
#     # 第三步：提取图像数据库返回的 id，并在文本数据库中查询对应的文本
#     image_ids = [result.id for result in image_search_results]
#     text_search_results = get_text_from_image_ids(image_ids)
#     # print(text_search_results)
#
#     # 第四步：根据检索到的结果生成提示词
#     prompt = generate_prompt(query_text, text_search_results)
#
#     # 第五步：调用大语言模型生成答案
#     answer = LLM_generate_answer(client, 'llava-med', prompt)
#
#     return answer

# def rag_pipeline(image_path, query_text):
#     image_vector = encode_image(image_path)
#     image_search_results = search_image_vectors(image_vector, top_k=20)
#     image_ids = [result['id'] for result in image_search_results]
#     text_search_results = get_text_from_image_ids(image_ids)
#     prompt = generate_prompt(query_text, text_search_results)
#     answer = llava_med_generate_answer(prompt)
#     return answer
#
# # 示例调用
# image_path = "../extra_50_0003.jpg"
# query_text = "对用户提供的SPECT影像数据，详细分析左右肾脏的健康状况..."
# answer = rag_pipeline(image_path, query_text)
# print("Answer:", answer)



# # 示例调用
# image_path = "../extra_50_0003.jpg"  # 输入图片路径
# query_text = ("对用户提供的SPECT影像数据，详细分析左右肾脏的健康状况，需要注意：The kidney on the left of the figure is the patient's actual left kidney, "
#               "and the kidney on the right is the patient's actual right kidney."
#               "针对每一侧肾脏，请根据影像特征，判断是否存在异常表现,如造影剂如何分布等情况。根据这些分析，做出对左右肾脏的健康评估")  # 输入查询文本
#
# answer = rag_pipeline(image_path, query_text)
# print("Answer:", answer)