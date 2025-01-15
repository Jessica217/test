import torch
import clip
from PIL import Image
from pymilvus import connections, MilvusClient
from milvus_model.hybrid import BGEM3EmbeddingFunction
from sentence_transformers import SentenceTransformer

class RAGPipeline:
    def __init__(self, milvus_uri="http://10.0.81.173:19530", device="cuda", use_fp16=False):
        """
        初始化 RAGPipeline。
        :param milvus_uri: Milvus 服务器地址。
        :param device: 使用的计算设备（如 "cuda" 或 "cpu"）。
        :param use_fp16: 是否使用 FP16 模式。
        """

        self.device = device if torch.cuda.is_available() else "cpu"
        self.client = MilvusClient(uri=milvus_uri)
        self.image_model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.text_model = None
        self.text_encoder = None
        self.use_fp16 = use_fp16

    def load_text_model(self, model_name_or_path):
        """
        加载文本嵌入模型。
        :param model_name_or_path: 文本嵌入模型的路径。
        """
        self.text_encoder = BGEM3EmbeddingFunction(
            model_name=model_name_or_path,
            device=self.device,
            use_fp16=self.use_fp16
        )

    def create_collection(self, collection_name, dimension):
        """
        创建一个 Milvus 集合。
        :param collection_name: 集合名称。
        :param dimension: 向量维度。
        """
        if collection_name not in self.client.list_collections():
            self.client.create_collection(
                collection_name=collection_name,
                dimension=dimension
            )
            print(f"Collection '{collection_name}' created with dimension {dimension}.")
        else:
            print(f"Collection '{collection_name}' already exists.")

    def encode_image(self, image_path):
        """
        编码图像为向量。
        :param image_path: 图像路径。
        :return: 图像向量。
        """

        image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_vector = self.image_model.encode_image(image).cpu().numpy().tolist()[0]
        return image_vector

    def encode_text(self, text):
        """
        编码文本为向量。
        :param text: 文本列表。
        :return: 文本向量。
        """
        text_vector = self.text_encoder.encode_documents(text)
        return text_vector['dense'][0].tolist()

    def insert_to_milvus(self, collection_name, data):
        """
        插入向量数据到 Milvus。
        :param collection_name: 集合名称。
        :param data: 数据，格式为 [{"id": id, "vector": vector, "text": text}, ...]
        """
        self.client.insert(collection_name=collection_name, data=data)
        print(f"Inserted data into collection '{collection_name}'.")

    def run_pipeline(self, image_path, text, text_model_path, image_collection_name, text_collection_name, image_id):
        """
        主流程：编码图像和文本，创建集合并插入数据。
        :param image_path: 图像路径。
        :param text: 文本描述。
        :param text_model_path: 文本模型路径。
        :param image_collection_name: 图像向量集合名称。
        :param text_collection_name: 文本向量集合名称。
        :param image_id: 数据唯一 ID。
        """
        # 加载文本模型
        self.load_text_model(text_model_path)

        # 编码图像和文本
        image_vector = self.encode_image(image_path)
        text_vector = self.encode_text([text])

        # 创建 Milvus 集合
        self.create_collection(image_collection_name, dimension=len(image_vector))
        self.create_collection(text_collection_name, dimension=len(text_vector))

        # 插入图像向量
        image_data = [{"id": image_id, "vector": image_vector, "text": image_path}]
        self.insert_to_milvus(image_collection_name, image_data)

        # 插入文本向量
        text_data = [{"id": image_id, "vector": text_vector, "text": text}]
        self.insert_to_milvus(text_collection_name, text_data)

        print("Pipeline completed successfully.")

# 使用示例
if __name__ == "__main__":
    # 参数配置
    MILVUS_URI = "http://10.0.81.173:19530"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    TEXT_MODEL_PATH = "/data/huggingface_models/bge-m3"
    IMAGE_COLLECTION_NAME = "wjy_image_embedding_collection"
    TEXT_COLLECTION_NAME = "wjy_text_embedding_collection"
    IMAGE_PATH = "0003.jpg"
    TEXT = "这是一张左肾脏是normal的，但是右肾脏是abnormal的，并且右侧的异常原因是肾脏过大"
    IMAGE_ID = 2

    # 初始化并运行流程
    pipeline = RAGPipeline(milvus_uri=MILVUS_URI, device=DEVICE)
    pipeline.run_pipeline(
        image_path=IMAGE_PATH,
        text=TEXT,
        text_model_path=TEXT_MODEL_PATH,
        image_collection_name=IMAGE_COLLECTION_NAME,
        text_collection_name=TEXT_COLLECTION_NAME,
        image_id=IMAGE_ID
    )
