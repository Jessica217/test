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
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images
from tokenizers import AddedToken
from llava.conversation import conv_templates
import argparse

#连接到 Milvus
connections.connect(uri="http://localhost:19530")

clip_model, clip_preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

image_collection_name = 'wjy_image_embedding_collection'  # 图像向量数据库集合名称
text_collection_name = 'wjy_text_embedding_collection'  # 文本向量数据库集合名称

# 获取集合对象
image_collection = Collection(image_collection_name)
text_collection = Collection(text_collection_name)

#设备设置：如果有GPU，则使用GPU
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
clip_model.to(device)
clip_model.eval()


# 使用biomed-clip对图像编码
def encode_image(image_path):
    raw_image = Image.open(image_path).convert('RGB')  # 打开图像
    image_feature = clip_preprocess(raw_image).unsqueeze(0).to(device)
    image_result = clip_model.encode_image(image_feature)  # 转换为PyTorch tensor并送入GPU
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
        print(matches)
        return matches
    else:
        print("No results found.")
        return []


def get_RAG_text_from_image_ids(image_ids):
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

    print(f"====================RAG_results==========================: {results}")
    return results


### 对召回结果引入投票机制

def vote_diagnosis(rag_results):
    """
    对 RAG 召回结果进行投票：
    - 针对左肾和右肾分别统计 'normal' 和 'abnormal' 的出现次数
    - 如果 abnormal 次数较多，则判定为 abnormal；否则为 normal
    - 根据医学规则，只要有一侧肾为 abnormal，最终诊断为 APN（Yes）

    参数:
        rag_results: List[dict]，每个字典包含键 "text" (诊断文本) 和 "id"

    返回:
        dict: 包含最终左右肾状态和整体诊断结果，同时附带各项计数信息
    """
    left_normal_count = 0
    left_abnormal_count = 0
    right_normal_count = 0
    right_abnormal_count = 0

    for item in rag_results:
        text = item.get("text", "").lower()
        # 将文本按句子拆分（简单用句点拆分）
        sentences = text.split(".")

        # 针对左肾
        left_found = False
        for sentence in sentences:
            if "left kidney" in sentence:
                # 如果句子同时包含 "normal" 和 "abnormal"，则不计入
                if "normal" in sentence and "abnormal" not in sentence:
                    left_normal_count += 1
                    left_found = True
                    break
                elif "abnormal" in sentence:
                    left_abnormal_count += 1
                    left_found = True
                    break
        # 如果找不到明确描述，可以选择不计入或者统计为 "unable to determine"

        # 针对右肾
        right_found = False
        for sentence in sentences:
            if "right kidney" in sentence:
                if "normal" in sentence and "abnormal" not in sentence:
                    right_normal_count += 1
                    right_found = True
                    break
                elif "abnormal" in sentence:
                    right_abnormal_count += 1
                    right_found = True
                    break
        # 同样，如果没有明确描述可选择跳过

    # 根据计数结果确定左右肾状态：
    left_final = "normal" if left_normal_count >= left_abnormal_count else "abnormal"
    right_final = "normal" if right_normal_count >= right_abnormal_count else "abnormal"

    # 诊断规则：只要任一侧为 abnormal，则最终诊断为 APN（Yes）
    overall_diagnosis = "Yes" if left_final == "abnormal" or right_final == "abnormal" else "No"

    return {
        "left_kidney": left_final,
        "right_kidney": right_final,
        "diagnosis": overall_diagnosis,
        "details": {
            "left_normal_count": left_normal_count,
            "left_abnormal_count": left_abnormal_count,
            "right_normal_count": right_normal_count,
            "right_abnormal_count": right_abnormal_count
        }
    }


def generate_prompt(query_text, RAG_search_results):
    # 生成提示词，结合图像和文本查询的结果
    # context = " ".join([result['text'] for result in RAG_search_results])  # 直接访问字典中的 'text' 字段
    prompt = f"""
You are a nephrology expert specializing in the diagnosis of acute pyelonephritis (APN) and have expertise in medical imaging analysis, particularly SPECT imaging.

### Task Description:

Your task is to analyze the provided SPECT medical images and corresponding medical text information ({RAG_search_results}) while strictly adhering to the following principles:
1.Base your responses solely on the provided supporting information. Do not add any content beyond the given information.
2.Prioritize using the exact wording from the supporting information to ensure accuracy.
2.Ensure that your response includes all key information required by the question, leaving no critical medical details omitted.

Please answer the following question: {query_text}


### Processing Requirements:

1. **Analysis of Left and Right Kidney Health Conditions**:
   -Evaluate the health status of both the left and right kidneys based on SPECT imaging.
   -Important note: The kidney on the left side of the image represents the patient's actual left kidney, and the right side represents the patient's actual right kidney.
   -Perform a detailed analysis, considering radioactive distribution, morphological changes, and other relevant characteristics.
   -If there is insufficient evidence to determine abnormalities, explicitly state "unable to determine" or "unable to draw a conclusion based on the available information."

2. **Acute Pyelonephritis Diagnosis**:
   -Based on SPECT imaging characteristics, if at least one kidney exhibits abnormalities, the patient is diagnosed with acute pyelonephritis (Yes).
   -If both kidneys appear normal, the patient is determined not to have APN (No).
   -If the available information is insufficient for a definitive diagnosis, explicitly state "unable to determine" or "unable to draw a conclusion based on the available information."

3. **Important Notes**:
   - The analysis must be based solely on the imaging data and the information retrieved from the RAG database. External assumptions or unverified information should not be included.
   - Clearly describe the conditions of each kidney, avoiding vague or ambiguous statements.
   - If the available information is insufficient for a definitive conclusion, explicitly state that the determination cannot be made.

4. **Final Output Format**:
  - **Left Kidney Assessment** (Normal or Abnormal, with details)
  - **Right Kidney Assessment** (Normal or Abnormal, with details)
  - **Acute Pyelonephritis Diagnosis** (Yes / No / Unable to determine)
"""


    return prompt


def llava_med_generate_answer(image_path, prompt):
    llava_tokenizer, llava_model, llava_image_processor, llava_context_len = load_pretrained_model(
        model_path="D:\\new_start\\llava-med",
        model_base=None,
        model_name='llava-med-v1.5-mistral-7b'
    )

    llava_tokenizer.add_tokens([AddedToken(DEFAULT_IMAGE_TOKEN, single_word=False, lstrip=True, rstrip=True, special=True)])

    # 读取图像
    image_data = Image.open(image_path).convert("RGB")
    image_tensor = llava_image_processor.preprocess(image_data, return_tensors='pt')['pixel_values'].half().cuda()

    # 回答模板
    conv_mode = 'llava_v1'
    conv = conv_templates[conv_mode].copy()

    if conv is not None and hasattr(conv, "copy"):
        conv = conv.copy()
        inp = DEFAULT_IMAGE_TOKEN + '\n' + prompt
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        final_prompt = conv.get_prompt()
    else:
        final_prompt = DEFAULT_IMAGE_TOKEN + '\n' + prompt

    input_ids = tokenizer_image_token(final_prompt, llava_tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    # 进行推理
    with torch.inference_mode():
        output_ids = llava_model.generate(
            inputs=input_ids,
            images=image_tensor,
            do_sample=True,
            image_sizes=[image_data.size],
            max_new_tokens=1024,
            temperature=0.2,
            use_cache=False
        )
    outputs = llava_tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    return outputs


if __name__ == "__main__":
    # 可选：如果需要通过命令行传入参数，可以使用 argparse，否则直接指定
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-path", type=str, default="../extra_50_0002.jpg", help="Path to the SPECT image")
    parser.add_argument("--query-text",
        type=str,
        default="Analyze the provided SPECT imaging data to evaluate the health status of both kidneys. "
                "Provide a detailed assessment for each kidney, identifying any potential abnormalities, "
                "disease indications, or signs of dysfunction.",
        help="Query to analyze kidney pathology")

    args = parser.parse_args()
    image_vector = encode_image(args.image_path)
    image_search_results = search_image_vectors(image_vector, top_k=20)
    image_ids = [result['id'] for result in image_search_results]
    text_search_results = get_RAG_text_from_image_ids(image_ids)

    filtered_result = vote_diagnosis(text_search_results)
    print("Voting Result:", filtered_result)

    prompt = generate_prompt(args.query_text, filtered_result)
    print("=========================prompt============================", prompt)

    answer = llava_med_generate_answer(args.image_path, prompt)

    print("=========================LLava-med===============================", answer)