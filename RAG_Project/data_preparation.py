#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Introduce :
@File      : data_preparation.py
@Time      : 2024/8/13 9:50
@Author    : wei.xu
@Tel       : 
@Email     : wei.xu@tophant.com
@pip       : pip install 
"""
import sys
import os
import time
import datetime
import numpy as np
import pandas as pd
import warnings
import json
from pathlib import Path
# 打印当前工作目录
current_directory = os.getcwd()
print("当前工作目录是：", current_directory)

#总的来说，这段代码是在帮助Python找到并使用存放在不同地方的代码文件，就像是给Python一个指南针，指引它在需要的时候能找到正确的路径。
warnings.simplefilter("ignore")
# 显示所有列
pd.set_option('display.max_columns', 20)
# 显示所有行
pd.set_option('display.max_rows', 50)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 100)
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
parentPath = os.path.split(rootPath)[0]
grandFatherPath = os.path.split(parentPath)[0]
sys.path.append(curPath)
sys.path.append(rootPath)
sys.path.append(parentPath)
sys.path.append(grandFatherPath)

import pandas as pd
from transformers import pipeline
import torch
from langchain_community.document_loaders import PyPDFLoader
from tqdm import tqdm
import re
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter

def clean_dataset(entire_text):
    text = entire_text.page_content
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.compile(r'(\w+)\s*-\s*(\w+)').sub(lambda match: match.group(1) + match.group(2), text)
    text = re.sub(r'^Investors’ Handbook\s*(?:\d+|[IVXLCDM]+)\s*', '', text, flags=re.IGNORECASE)
    entire_text.page_content = text
    return entire_text

# Loading the pdf of NEPSE booklet
# pdf_name = "NEPSE_booklet.pdf"
pdf_name = "网络安全词汇术语汇编 - v8.0(国标纪念版).pdf"
loader = PyPDFLoader(pdf_name)
pages = loader.load_and_split()
print(len(pages))


page_numbers = list(range(5, len(pages)))
page_numbers = sorted(page_numbers)
pages1 = [pages[i] for i in page_numbers]
pages1 = [clean_dataset(i.model_copy()) for i in tqdm(pages1, desc="Cleaning Pages")]
print(len(pages1))

text_splitter = CharacterTextSplitter(
    chunk_size = 5000,
    chunk_overlap = 150,
    separator="."
)

docs = text_splitter.split_documents(pages1)
print(len(docs))
print(f"*********************************")
print(docs[0].page_content)
print(f"*********************************")
print(docs[1].page_content)
print(f"*********************************")

#
model_name='/data/huggingface_models/bge-m3'
# model_name='/data/huggingface_models/bge-large-zh-v1.5'
# model_name='/data/huggingface_models/acge_text_embedding'
# model_name='/data/huggingface_models/all-mpnet-base-v2'
sentence_transformer = HuggingFaceEmbeddings(
    model_name=model_name,# 'sentence-transformers/all-MiniLM-L6-v2',
    model_kwargs={'device':'cuda'},
    encode_kwargs={'normalize_embeddings': True}
)

# creating and storing embeddings generated via sentence_transformer in the FAISS vector store
vector_db = FAISS.from_documents(docs, sentence_transformer)
vector_db_name="vector_db_cyber_sec_words"
vector_db.save_local(vector_db_name)
docsearch = FAISS.load_local(vector_db_name, sentence_transformer,allow_dangerous_deserialization=True)


while True:
    query=input("输入你要查询的:")
    if not query:
        continue
    elif query=='break':
        break
    else:
        results = vector_db.similarity_search_with_score(
            query, k=10
        )
        for res, score in results:
            print(f"* [SIM={score:3f}] {res.page_content} [{res.metadata}]")

