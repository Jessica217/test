from transformers import BlipProcessor, BlipForConditionalGeneration
from pymilvus import connections, Collection
import numpy as np
import requests
import json
from openai import OpenAI
from PIL import Image
import torch



# 加载 BLIP 模型和处理器
model_path = '/data/wjy/blip-image-captioning-large'  # 你可以根据需要替换成其他预训练模型
processor = BlipProcessor.from_pretrained(model_path)
image_model = BlipForConditionalGeneration.from_pretrained(model_path)

# 连接 Milvus 图像向量数据库
milvus_host = '10.0.81.173'  # Milvus服务地址
milvus_port = '19530'  # Milvus服务端口
image_collection_name = 'wjy_image_embedding_collection'  # 图像向量数据库集合名称
text_collection_name = 'wjy_text_embedding_collection'  # 文本向量数据库集合名称

connections.connect("default", host=milvus_host, port=milvus_port)
image_collection = Collection(image_collection_name)
text_collection = Collection(text_collection_name)

# 设置大语言模型的API

client = OpenAI(
    api_key="sk-a6795bbb3131498d8d1dedd46b04a2a0",
    # api_key=config['OPENAI']['deepseek_API_KEY'],
    #base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", # qwen url
    # base_url="https://open.bigmodel.cn/api/paas/v4/", # glm4 url
    base_url="https://api.deepseek.com/v1", # deep seek url
)

# 确保设备是GPU（如果可用），否则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 将模型移动到设备上
image_model = image_model.to(device)
# BLIP模型将图片编码为向量
def encode_image(image_path):
    # 处理图片
    raw_image = Image.open(image_path).convert('RGB')
    image_resized = raw_image.resize((300, 300))
    image_inputs = processor(image_resized, return_tensors="pt").to(device)  # 转换为PyTorch tensor并送入GPU
    # 图片编码
    with torch.no_grad():
        out = image_model.vision_model(**image_inputs).last_hidden_state
        # out = image_model.vision_model(**image_inputs).pooler_output
    image_vectors = out[:, 0, :].detach().cpu().numpy().tolist() # pooler_output
    image_vectors = image_vectors[0]
    print(image_vectors)

    return image_vectors


# 检索与图像查询向量最相似的前20个向量
def search_image_vectors(query_vector, top_k=20):
    query_vector = np.array([query_vector], dtype=np.float32)
    search_params = {
        "metric_type": "COSINE",
        "params": {'nprobe': 20, 'level': 3, 'radius': 0.8, 'range_filter': 1}  # radius是返回余弦相似度分数在0.8以内的搜索结果
    }
    # 查询图像向量数据库
    search_results = image_collection.search(
        query_vector,  # 查询向量
        anns_field="vector",  # 向量字段名
        param=search_params,  # 搜索参数，可以调整
        limit=top_k  # 返回前top_k个结果
    )
    # print(search_results[0])

    # 返回匹配的top_k结果
    return search_results[0]  # 返回的是一个列表，包含前top_k个检索结果

# 根据图像数据库中的id找到对应的文本描述
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



# 生成提示词
def generate_prompt(query_text, text_search_results):
    # 生成提示词，结合图像和文本查询的结果
    context = " ".join([result['text'] for result in text_search_results])  # 直接访问字典中的 'text' 字段
    prompt = f"""
    ### 任务描述：
基于用户上传的影像数据 {context}，请回答：{query_text}。

### 处理要求：

1. **左右肾脏健康状况分析**：  
   根据SPECT影像分析结果，对左右肾脏的健康状况做出评估。请注意：  
   - 输入图像的**左侧肾脏即为病人的实际左肾**，**右侧肾脏即为病人的实际右肾**。
   - 请分别分析左侧肾脏和右侧肾脏的正常与异常情况，包括放射性分布、形态变化等特征。
   - 如果影像中没有足够的证据或无法确定异常，请明确说明“无法判断”或“根据现有信息无法得出结论”。

2. **急性肾盂肾炎诊断**：  
   在完成左右肾脏健康状况分析后，基于影像中的肾脏特征，判断患者是否存在**急性肾盂肾炎**的可能。如果无法从影像中确定或没有足够的证据，请明确说明“无法判断”或“根据现有信息无法得出结论”。

3. **输出格式要求**：
   - **左右肾脏的健康评估**：对于每一侧肾脏（左侧、右侧），请分别分析并给出**正常或异常**的情况。
   - **急性肾盂肾炎的诊断结论**：给出结论，是否符合急性肾盂肾炎的诊断标准（是/否/无法判断）。
   - 请确保结论基于影像信息，不要添加未经证实的推论。

4. **注意事项**：
   - 仅依据影像数据和RAG数据库中的信息进行分析，不可添加外部信息或假设。
   - 确保清晰地表述每侧肾脏的具体情况，避免模糊或不明确的回答。
   - 如果影像提供的信息不足以做出明确结论，务必说明无法判断或根据现有信息无法得出结论。

**最终输出格式**：
- **左侧肾脏**的评估（正常或异常，具体表现）  
- **右侧肾脏**的评估（正常或异常，具体表现）  
- **急性肾盂肾炎的诊断结果**（是/否/无法判断）
"""
    return prompt


# 调用大语言模型生成答案
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


# 流程
def rag_pipeline(image_path, query_text):
    # 第一步：将图片编码成向量
    image_vector = encode_image(image_path)

    # 第二步：在图像数据库中检索与图像查询向量最相似的前20个图像向量
    image_search_results = search_image_vectors(image_vector, top_k=20)

    # 第三步：提取图像数据库返回的 id，并在文本数据库中查询对应的文本
    image_ids = [result.id for result in image_search_results]
    text_search_results = get_text_from_image_ids(image_ids)
    # print(text_search_results)

    # 第四步：根据检索到的结果生成提示词
    prompt = generate_prompt(query_text, text_search_results)

    # 第五步：调用大语言模型生成答案
    answer = LLM_generate_answer(client, 'deepseek-chat', prompt)

    return answer


# 示例调用
image_path = "extra_50_0003.jpg"  # 输入图片路径
query_text = ("对用户提供的SPECT影像数据，详细分析左右肾脏的健康状况，需要注意：图中的左侧肾脏即为患者的左侧肾脏，右侧肾脏即为患者的右侧肾脏。"
              "针对每一侧肾脏，请根据影像特征，判断是否存在异常表现,如造影剂如何分布等情况。根据这些分析，做出对左右肾脏的健康评估")  # 输入查询文本

answer = rag_pipeline(image_path, query_text)
print("Answer:", answer)
