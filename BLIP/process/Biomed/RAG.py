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
import argparse
from conversation import Conversation, SeparatorStyle
from transformers import AutoProcessor
from llava.conversation import conv_templates, SeparatorStyle


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


llava_tokenizer, llava_model, llava_image_processor, llava_context_len = load_pretrained_model(
        model_path="D:\\new_start\\llava-med",
        model_base=None,
        model_name='llava-med-v1.5-mistral-7b'
    )

llava_tokenizer.add_tokens([AddedToken(DEFAULT_IMAGE_TOKEN, single_word=False, lstrip=True, rstrip=True, special=True)])

# 使用biomed-clip对图像编码
def encode_image(image_path):
    raw_image = Image.open(image_path).convert('RGB')  # 打开图像
    image_feature = clip_preprocess(raw_image).unsqueeze(0).to(device)
    image_result = clip_model.encode_image(image_feature)  # 转换为PyTorch tensor并送入GPU
    # 确保在转换之前移除梯度信息
    image_vectors = image_result[0].detach().cpu().numpy().tolist()

    return image_vectors


# 图像向量查询函数
def search_image_vectors(query_vector, top_k):
    query_vector = np.array([query_vector], dtype=np.float32)
    search_params = {
        "metric_type": "COSINE",
        "params": {'nprobe': 10, 'level': 3, 'radius': 0.8, 'range_filter': 1}
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
            text_data = text_collection.query(expr=f"id == {result.id}", output_fields=["text"])
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
        result = text_collection.query(expr=f"id == {image_id}", output_fields=["text", "id"])
        if result:
            data = result[0] if isinstance(result[0], dict) else eval(result[0])
            results.append({'text': data.get('text', 'No text available'), 'id': data.get('id', image_id)})
    return results


def generate_prompt(query_text, RAG_search_results):
    # 生成提示词，结合图像和文本查询的结果
    context = " ".join([result['text'] for result in RAG_search_results])  # 直接访问字典中的 'text' 字段
    prompt = f"""
You are a nephrology expert specializing in the diagnosis of acute pyelonephritis (APN) and have expertise in medical imaging analysis, particularly SPECT imaging.

These are the common methods and basis for radiologists to evaluate DMSA renal imaging. You are a nephrology expert, You must remember:
Normal renal static imaging: Both kidneys are bean-shaped, with clear images and intact contours. The outer zones of the renal shadows show higher radioactivity concentration, while the central and hilar areas are slightly lighter, with no significant difference in radioactive distribution between the two kidneys.
Abnormal imaging: Various diseases can cause localized or generalized renal function impairment, which may manifest as abnormalities in kidney position, morphology, or number; localized sparse or absent radioactive distribution; localized increased radioactivity; or faint or non-visualized renal shadows.

### Task Description:

Your task is to analyze the provided SPECT medical images and corresponding medical text information ({context}) while strictly adhering to the following principles:
1.Base your responses solely on the provided supporting information. Do not add any content beyond the given information.
2.Prioritize using the exact wording from the supporting information to ensure accuracy.
2.Ensure that your response includes all key information required by the question, leaving no critical medical details omitted.

Please answer the following question: {query_text}.

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

    # 读取图像
    image_data = Image.open(image_path).convert("RGB")
    image_tensor = llava_image_processor.preprocess(image_data, return_tensors='pt')['pixel_values'].half().cuda()

    conv = conv_templates['llava_v1'].copy()
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + prompt)
    conv.append_message(conv.roles[1], None)

    # print("final_prompt", final_prompt)
    input_ids = tokenizer_image_token(conv.get_prompt(), llava_tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    # 进行推理
    with torch.inference_mode():
        output_ids = llava_model.generate(
            inputs=input_ids,
            images=image_tensor,
            do_sample=True,
            image_sizes=[image_data.size],
            max_new_tokens=1024,
            temperature=0.1,
            use_cache=False
        )
    outputs = llava_tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    return outputs


def process_images_in_folder(image_folder, query_text, output_file):
    results = []
    for image_name in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_name)
        if not image_path.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue

        print(f"Processing: {image_name}")
        image_vector = encode_image(image_path)
        image_search_results = search_image_vectors(image_vector, top_k=5)
        image_ids = [result['id'] for result in image_search_results]
        text_search_results = get_RAG_text_from_image_ids(image_ids)
        prompt = generate_prompt(query_text, text_search_results)
        answer = llava_med_generate_answer(image_path, prompt)

        results.append({"image": image_path, "answer": answer})
        print(f"Finished: {image_name}\n")

    # 保存结果
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    # 可选：如果需要通过命令行传入参数，可以使用 argparse，否则直接指定
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-folder", type=str, default="D:/maskrcnn/datasets/extra_DMSA_VAL/images", help="Folder containing SPECT images")

    # parser.add_argument("--image-path", type=str, default="D:/maskrcnn/datasets/extra_DMSA_VAL/RGB_mode/0004.jpg", help="Path to the SPECT image")
    parser.add_argument("--query-text",
        type=str,
        default="Analyze the provided SPECT imaging data to evaluate the health status of both kidneys. "
                "Provide a detailed assessment for each kidney, identifying any potential abnormalities, disease indications, or signs of dysfunction."
                "State any abnormalities, disease indicators, or dysfunctions concisely. Do not generate explanations—only give the final medical assessment.",
        help="Query to analyze kidney pathology")

    parser.add_argument("--output-file", type=str, default="RAG_output_topk5.json", help="File to save results")


    args = parser.parse_args()
    process_images_in_folder(args.image_folder, args.query_text, args.output_file)
