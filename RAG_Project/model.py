import time
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from FlagEmbedding import FlagModel
import requests, re, urllib.parse, torch
from threading import Thread
from FlagEmbedding import FlagReranker, FlagLLMReranker


class PredictionPipeline:
    def __init__(self,model_type='FlagLLMReranker',use_fp16=True, use_bf16=False):
        print(f"model_type: {model_type}")
        self.model_type = model_type
        # self.model_id = "/data/huggingface_models/Qwen2-7B-Instruct"  # 'TheBloke/Starling-LM-7B-alpha-GPTQ'
        self.model_id = "/data/huggingface_models/safe_v3_checkpoint60"  # 'TheBloke/Starling-LM-7B-alpha-GPTQ'
        self.temperature = 0.7
        # self.bit = ["gptq-4bit-32g-actorder_True", "gptq-8bit-128g-actorder_True"]
        # self.sentence_transformer_modelname = 'sentence-transformers/all-mpnet-base-v2' # 'sentence-transformers/all-MiniLM-L6-v2'
        # self.sentence_transformer_modelname = '/data/huggingface_models/bge-large-zh-v1.5' # 'sentence-transformers/all-MiniLM-L6-v2'
        # self.sentence_transformer_modelname = '/data/huggingface_models/all-mpnet-base-v2' # 'sentence-transformers/all-MiniLM-L6-v2'
        # self.sentence_transformer_modelname = '/data/huggingface_models/bge-m3'  # 'sentence-transformers/all-MiniLM-L6-v2'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = "auto"
        self.torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16
        self.vector_db_name = "vector_db_cyber_sec_words"
        # self.reranking_model_name = '/data/huggingface_models/bge-large-zh-v1.5'
        # self.llm_reranking_model_name = '/data/huggingface_models/bge-reranker-v2-gemma'
        # self.llm_reranking_model_name = '/data/huggingface_models/bge-reranker-v2-m3'
        # 初始化重排序模型，根据传入的 model_type 初始化不同的模型


        print(f"1. Device being utilized: {self.device} !!!")

    def load_model_and_tokenizers(self):
        '''
        This method will initialize the tokenizer and our LLM model and the streamer class.
          该方法将初始化分词器、我们的LLM模型以及流式处理类。
        '''
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, torch_dtype=self.torch_dtype,
                                                       device_map=self.device, use_fast=True, model_max_length=4000)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, device_map=self.device,
                                                          trust_remote_code=False,
                                                          torch_dtype=self.torch_dtype,
                                                          # revision=self.bit[1]
                                                          )
        self.streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True)
        # print(f'2. {self.model_id} has been successfully loaded !!!')
        print(f'2. {self.model_id} 已成功加载 !!!')

    def load_sentence_transformer(self):
        '''
        This method will initialize our sentence transformer model to generate embeddings for a given query.
            该方法将初始化我们的句子转换器模型，以生成给定查询的嵌入。
        '''
        self.sentence_transformer = HuggingFaceEmbeddings(
            model_name=self.sentence_transformer_modelname,
            model_kwargs={'device': 'cuda'},
            encode_kwargs={'normalize_embeddings': True}
        )
        # print("3. Sentence Transformer Loaded !!!!!!")
        print("3. Sentence Transformer已加载 !!!!!!")

    def load_reranking_model(self):
        '''
        An opensoure reranking model called bge-reranker from huggingface is utilized to perform reranking on the retrived relevant documents from vector store.
        This method will initialize the reranking model.
        一个名为 bge-reranker 的开源重排序模型来自 Huggingface，用于对从向量存储中检索到的相关文档进行重排序。
        该方法将初始化重排序模型。
        '''
        # self.reranker = FlagModel(self.reranking_model_name,use_fp16=True)  # 'BAAI/bge-reranker-large'->2GB BAAI/bge-reranker-base-> 1GB

        if self.model_type == 'FlagLLMReranker':
            # FlagLLMReranker 对应 /data/huggingface_models/bge-reranker-v2-gemma
            self.reranker = FlagLLMReranker('/data/huggingface_models/bge-reranker-v2-gemma', use_bf16=True)
        elif self.model_type == 'FlagReranker':
            # FlagReranker 对应 /data/huggingface_models/bge-reranker-v2-m3
            self.reranker = FlagReranker('/data/huggingface_models/bge-reranker-v2-m3', use_fp16=True)
        elif self.model_type == 'FlagModel':
            # FlagModel 对应 /data/huggingface_models/bge-large-zh-v1.5
            self.reranker = FlagModel('/data/huggingface_models/bge-large-zh-v1.5', use_fp16=True)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
        print(f"Reranker initialized with model type: {self.model_type}")
        # print("4. Re-Ranking Algorithm Loaded !!!")
        print("4. 重排序算法已加载 !!!")

    def load_llm_reranking_model(self):
        '''
        An opensoure reranking model called bge-reranker from huggingface is utilized to perform reranking on the retrived relevant documents from vector store.
        This method will initialize the reranking model.
        一个名为 bge-reranker 的开源重排序模型来自 Huggingface，用于对从向量存储中检索到的相关文档进行重排序。
        该方法将初始化重排序模型。
        '''
        self.reranker = FlagModel(self.reranking_model_name,
                                  use_fp16=True)  # 'BAAI/bge-reranker-large'->2GB BAAI/bge-reranker-base-> 1GB
        print("4. Re-Ranking Algorithm Loaded !!!")
        print("4. 重排序算法已加载 !!!")


    def load_embeddings(self):
        '''
        This method will load the FAISS vector database that was developed in the Data_prerpation_NEPSE. 
         该方法将加载在 Data_prerpation_NEPSE 中开发的 FAISS 向量数据库。
        '''
        self.vector_db = FAISS.load_local(self.vector_db_name, self.sentence_transformer,
                                          allow_dangerous_deserialization=True)
        print(f"5. FAISS VECTOR STORE LOADED !!!")
        print(f"5. FAISS 向量存储已加载 !!!")

    def rerank_contexts(self, query, contexts, number_of_reranked_documents_to_select=5):
        '''
        对检索到的文档进行重新排序。

        参数:
        query -> 用户提出的问题
        contexts -> 从向量存储中检索到的相关文档
        number_of_reranked_documents_to_select -> 重新排序后选择的前 k 个文档。

        返回:
        重新排序后的前 k 个上下文。 [列表]
        '''
        start = time.time()

        if self.model_type in ['FlagReranker', 'FlagLLMReranker']:
            # 计算查询与每个上下文的相似度得分
            similarity_scores = []
            for context in contexts:
                score = self.reranker.compute_score([query, context], normalize=True)
                similarity_scores.append(score[0])
                print(f"Computed similarity score for context: {context}, \n score: {score[0]}")
            print(f"similarity_scores:{similarity_scores}")
        elif self.model_type == 'FlagModel':
            # 使用重新排序器的嵌入模型对查询和上下文进行编码
            print(f"query: {query}")
            print("使用重新排序器的嵌入模型对查询进行编码")
            embeddings_1 = self.reranker.encode(query)
            print("使用重新排序器的嵌入模型对上下文进行编码")
            print(f"contexts:\n{contexts}")
            embeddings_2 = self.reranker.encode(contexts)
            # 计算查询与每个上下文之间的相似度
            print("计算查询与每个上下文之间的相似度")
            similarity_scores = embeddings_1 @ embeddings_2.T
            print(f"similarity_scores: {similarity_scores}")
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        # 确保选择的重新排序文档数量不超过上下文的总数。
        number_of_contexts = len(contexts)
        if number_of_reranked_documents_to_select > number_of_contexts:
            print(
                f"警告！！！上下文的长度 ({number_of_contexts}) 小于要选择的重新排序文档数量 ({number_of_reranked_documents_to_select})")
            number_of_reranked_documents_to_select = number_of_contexts

        # 根据相似度选择最高排名的上下文的索引
        print("根据相似度选择最高排名的上下文")
        highest_ranked_indices = sorted(range(len(similarity_scores)), key=lambda i: similarity_scores[i],
                                        reverse=True)[:number_of_reranked_documents_to_select]
        print(f"highest_ranked_indices: {highest_ranked_indices}")

        # 根据选定的索引返回重新排序后的上下文
        print("返回重新排序后的上下文")
        reranked_contexts = [contexts[index] for index in highest_ranked_indices]

        end = time.time()
        print(f"total_time: {(end - start)} seconds")

        return reranked_contexts

    def is_text_nepali(self, text):
        '''
        这个方法用于检查用户提出的问题中是否包含任何尼泊尔语单词。如果包含，LLM（大语言模型）的响应也会使用谷歌翻译API返回尼泊尔语。

        参数:
        text -> 用户提出的问题

        返回值: bool
        如果文本中包含尼泊尔语单词，则返回True，否则返回False
        '''
        nepali_regex = re.compile(r'[\u0900-\u097F]+')
        if nepali_regex.search(text):
            return True
        return False

    def translate_using_google_api(self, text, source_language="auto", target_language="ne", timeout=5):
        '''
        This function has been copied from here:
        # https://github.com/ahmeterenodaci/easygoogletranslate/blob/main/easygoogletranslate.py

        This free API is used to perform translation between English to Nepali and vice versa.

        parameters: 
        source_language -> the language code for the source language
        target_language -> the new language to which the text is to be translate 

        returns
        '''
        pattern = r's(?s)clas="(?:t0|result-container)">(.*?)<'
        escaped_text = urllib.parse.quote(text.encode('utf8'))
        url = 'https://translate.google.com/m?tl=%s&sl=%s&q=%s' % (target_language, source_language, escaped_text)
        response = requests.get(url, timeout=timeout)
        result = response.text.encode('utf8').decode('utf8')
        result = re.findall(pattern, result)
        return result

    def split_and_translate_text(self, text, source_language="auto", target_language="ne", max_length=5000):
        """
        Split the input text into sections with a maximum length.
        
        Parameters:
        - text: The input text to be split.
        - max_length: The maximum length for each section (default is 5000 characters).

        Returns:c
        A list of strings, each representing a section of the input text.
        """

        if source_language == "en":
            splitted_text = text.split(".")
        elif source_language == "ne":
            splitted_text = text.split("।")
        else:
            splitted_text = [text[i:i + max_length] for i in range(0, len(text), max_length)]

        # perform translation (the free google api can only perform translation for 5000 characters max. So, splitting the text is necessary )
        translate_and_join_splitted_text = " ".join(
            [self.translate_using_google_api(i, source_language, target_language)[0] for i in splitted_text])
        return translate_and_join_splitted_text

    def perform_translation(self, question, source_language, target_language):
        try:
            # Check if the length of the question is greater than 5000 characters
            if len(question) > 5000:
                # If so, split and translate the text using a custom method
                return self.split_and_translate_text(question, source_language, target_language)
            else:
                # If not, use the Google Translation API to translate the entire text
                return self.translate_using_google_api(question, source_language, target_language)[0]
        except Exception as e:
            return [f"An error occurred, [{e}], while working with Google Translation API"]

    def make_predictions(self, question, top_n_values=10):
        '''
        该方法将执行预测操作
        参数：
        question -> 用户提出的问题
        top_n_values -> 从向量存储中选择的相关文档的前 n 个值
        This method will perform the prediction
        Parameters:
        question -> The question asked by the user
        top_n_values -> The top n values to select from the relavant retrived documents from vector store.
        '''

        # this method checks if the question asked by the user is nepali or not
        print(f"question:{question}")
        print(f"检查用户提出的问题是否是尼泊尔语")
        is_original_language_nepali = self.is_text_nepali(question)
        print(f"is_original_language_nepali:{is_original_language_nepali}")

        # if the text is nepali, translate it to english first to get relevant docs from vector store, else just extract relavant docs from vector store
        if is_original_language_nepali:
            question = self.perform_translation(question, 'ne', 'en')
            print("Translated Question: ", question)
            if isinstance(question, list):
                yield "data: " + str(question[0]) + "\n\n"
                yield "data: END\n\n"

        # get relevant docs from vector store with similarity score (l2 distance /euclidean distance)
        print(f"从向量存储中获取具有相似度分数的相关文档（L2 距离/欧几里得距离）")
        similarity_search = self.vector_db.similarity_search_with_score(question, k=top_n_values)
        context = []
        for doc, score in similarity_search:
            print(f"* [SIM={score:3f}] {doc.page_content} [{doc.metadata}]")
            # only select the relevant docs with euclidean distance less than 1.5
            if score < 1.5:
                print(f"只选择欧几里得距离小于 1.5 的相关文档")
                context.append(doc.page_content)

        # only select the relevant docs with euclidean distance less than 1.5
        # context = [doc.page_content for doc, score in similarity_search if score < 1.5]
        number_of_contexts = len(context)
        print(f"number_of_contexts: {number_of_contexts}")

        if number_of_contexts == 0:
            # yield "data: Please know that the question asked and domain knowledge provided are irrelavant. Therefore, unable to provide answer to this question. Thank you.\n\n"
            yield "data: 请注意，所提的问题与提供的领域知识无关，因此无法回答此问题。谢谢。\n\n"

        else:
            if number_of_contexts > 1:
                # perform reranking
                print(f"perform reranking")
                context = self.rerank_contexts(question, context)

            context = ". ".join(context)
            print(f"context:\n{context}")

            # the prompt being used to be passed into the LLM
            # prompt = f'''
            #         Based solely on the information given in the context above, answer the following question.
            #         Never answer a question in your own words outside of the context provided.
            #         If the information isn’t available in the context to formulate an answer, politely say "Sorry, I don’t have knowledge about that topic."
            #         Please do not provide additional explanations or information by answering outside of the context.
            #         Always answer in maximum five sentences and less than hundred words.
            #
            #         \n\n
            #         Question: {question}\n\n
            #         Context: {context}\n\n
            #         Answer:
            # '''
            prompt = f"""
                        基于上面提供的上下文信息，回答以下问题。
                        绝不要在上下文提供的信息之外用自己的话回答问题。
                        如果上下文中没有足够的信息来回答，请礼貌地说“抱歉，我没有关于该话题的知识。”
                        请不要在回答时提供额外的解释或信息，回答内容必须完全基于上下文。
                        始终在五句话以内回答，并且字数少于一百字。
                         \n\n
                        问题： {question}\n\n
                        上下文： {context}\n\n
                        回答：
                """
            print(f"prompt:\n{prompt}")

            # performing tokenization and passing input to GPU
            print(f"执行分词并将输入传递到 GPU")
            inputs = self.tokenizer([prompt], return_tensors="pt").to("cuda")
            generation_kwargs = dict(inputs, streamer=self.streamer, max_new_tokens=8000, do_sample=True,
                                     temperature=0.7,
                                     top_p=0.95,
                                     top_k=40,
                                     repetition_penalty=1.1, pad_token_id=50256)

            '''
            Since LLMs are auto-regressive models, they are able to predict the next word in sequence. This means, as the model keeps on predicting the next word-
            - we can access the word and pass to the front-end. This efficitively improves user experience as the user won't have to wait until an entire response has
            been generated. This is also called text/response streaming.
            
            Here, I use threading to get the tokens being generated in real-time and utilize SSE (Server side events) to stream the responses to frontend in real time.
            由于大型语言模型（LLM）是自回归模型，它们能够预测序列中的下一个词。
            这意味着，随着模型不断预测下一个词，我们可以即时获取这个词并传递给前端。
            这有效地改善了用户体验，因为用户不必等待整个回复生成完毕。这种技术也被称为文本/响应流式传输。
            在这里，我使用了多线程来实时获取生成的词，并利用服务器端事件（SSE）将这些响应流式传输到前端。
            '''
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)

            thread.start()
            if is_original_language_nepali:
                sentence = ""
                for token in self.streamer:
                    if token != "</s>":
                        sentence += token
                        if "." in token:
                            sentence = self.translate_using_google_api(sentence, "en", "ne")[0]
                            sentence = re.sub(r'</?s>', '', sentence)  # This will remove both <s> and </s> if present
                            yield f"data: {sentence}\n\n"  # Format for SSE
                            sentence = ""
            else:
                for token in self.streamer:
                    yield f"data: {token}\n\n"  # Format for SSE
            thread.join()
        yield "data: END\n\n"


if __name__ == '__main__':
    # Initialize a PredictionPipeline object
    pipeline = PredictionPipeline()
    pipeline.load_model_and_tokenizers()
    pipeline.load_sentence_transformer()
    pipeline.load_reranking_model()
    pipeline.load_embeddings()
