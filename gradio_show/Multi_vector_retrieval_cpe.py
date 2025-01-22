#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Introduce :
@File      : Multi_vector_retrieval.py
@Time      : 2024/11/18 15:30
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
from FlagEmbedding import BGEM3FlagModel

# 初始化模型
model_name = "/data/huggingface_models/bge-m3"
model = BGEM3FlagModel(
    model_name_or_path=model_name,
    use_fp16=True,
    pooling_method='cls',
    device='cuda:1'
)

# 查询 CPE
query_cpe = """
{
    "part": "a",
    "vendor": "citrix",
    "product": "ica_client",
    "version": "6.1",
    "update": "",
    "edition": "",
    "language": ""
}
"""

# 文档 CPE 数据 (20 个例子)
documents_cpe = [
    {
        "cpe": """{
            "part": "a",
            "vendor": "citrix",
            "product": "ica_client",
            "version": "6.1",
            "update": "",
            "edition": "",
            "language": ""
        }""",
        "relevance": 1  # 完全匹配，相关
    },
    {
        "cpe": """{
            "part": "a",
            "vendor": "citrix",
            "product": "ica_client",
            "version": "-",
            "update": "",
            "edition": "",
            "language": ""
        }""",
        "relevance": 0.8  # 部分匹配，版本信息缺失
    },
    {
        "cpe": """{
            "part": "a",
            "vendor": "citrix",
            "product": "ica_client",
            "version": "6.0",
            "update": "",
            "edition": "",
            "language": ""
        }""",
        "relevance": 0.7  # 部分匹配，版本稍不同
    },
    {
        "cpe": """{
            "part": "a",
            "vendor": "citrix",
            "product": "hdx_wmi_provider",
            "version": "2.0.0.0",
            "update": "",
            "edition": "",
            "language": ""
        }""",
        "relevance": 0.5  # 不完全匹配，产品不同
    },
    {
        "cpe": """{
            "part": "h",
            "vendor": "citrix",
            "product": "netscaler_access_gateway",
            "version": "-",
            "update": "-",
            "edition": "enterprise",
            "language": ""
        }""",
        "relevance": 0.3  # 不匹配，完全不同
    },
    # 添加更多 CPE 示例，总计 20 个
    {
        "cpe": """{
            "part": "a",
            "vendor": "citrix",
            "product": "xenapp_server_sdk",
            "version": "6.1.2.0",
            "update": "",
            "edition": "",
            "language": ""
        }""",
        "relevance": 0.4
    },
    {
        "cpe": """{
            "part": "a",
            "vendor": "citrix",
            "product": "application_delivery_controller",
            "version": "12.1-55.247",
            "update": "",
            "edition": "",
            "language": ""
        }""",
        "relevance": 0.3
    },
    {
        "cpe": """{
            "part": "h",
            "vendor": "citrix",
            "product": "application_delivery_controller",
            "version": "-",
            "update": "",
            "edition": "",
            "language": ""
        }""",
        "relevance": 0.2
    },
    {
        "cpe": """{
            "part": "a",
            "vendor": "microsoft",
            "product": "windows_server",
            "version": "2016",
            "update": "",
            "edition": "",
            "language": ""
        }""",
        "relevance": 0.1  # 完全不同厂商
    },
    {
        "cpe": """{
            "part": "a",
            "vendor": "citrix",
            "product": "receiver_hdx_flash_redirection",
            "version": "13.0.0.6684",
            "update": "",
            "edition": "",
            "language": ""
        }""",
        "relevance": 0.6
    }
]
additional_documents_cpe = [
    {
        "cpe": """{
            "part": "a",
            "vendor": "citrix",
            "product": "netscaler_application_delivery_controller",
            "version": "12.1",
            "update": "",
            "edition": "",
            "language": ""
        }""",
        "relevance": 0.9  # 几乎完全匹配，相关
    },
    {
        "cpe": """{
            "part": "a",
            "vendor": "citrix",
            "product": "netscaler_application_delivery_controller",
            "version": "11.1",
            "update": "",
            "edition": "",
            "language": ""
        }""",
        "relevance": 0.8  # 部分版本不同
    },
    {
        "cpe": """{
            "part": "a",
            "vendor": "citrix",
            "product": "receiver_hdx_flash_redirection",
            "version": "13.0.0.6685",
            "update": "",
            "edition": "",
            "language": ""
        }""",
        "relevance": 0.7
    },
    {
        "cpe": """{
            "part": "h",
            "vendor": "citrix",
            "product": "netscaler_access_gateway",
            "version": "-",
            "update": "-",
            "edition": "standard",
            "language": ""
        }""",
        "relevance": 0.6  # 不完全匹配，edition 不同
    },
    {
        "cpe": """{
            "part": "a",
            "vendor": "oracle",
            "product": "java_runtime_environment",
            "version": "8u231",
            "update": "",
            "edition": "",
            "language": ""
        }""",
        "relevance": 0.5
    },
    {
        "cpe": """{
            "part": "h",
            "vendor": "citrix",
            "product": "netscaler_gateway",
            "version": "-",
            "update": "-",
            "edition": "enterprise",
            "language": ""
        }""",
        "relevance": 0.4
    },
    {
        "cpe": """{
            "part": "a",
            "vendor": "ibm",
            "product": "db2",
            "version": "11.5",
            "update": "",
            "edition": "advanced",
            "language": ""
        }""",
        "relevance": 0.3
    },
    {
        "cpe": """{
            "part": "a",
            "vendor": "apache",
            "product": "http_server",
            "version": "2.4.46",
            "update": "",
            "edition": "",
            "language": ""
        }""",
        "relevance": 0.2
    },
    {
        "cpe": """{
            "part": "a",
            "vendor": "microsoft",
            "product": "windows_10",
            "version": "1909",
            "update": "",
            "edition": "",
            "language": ""
        }""",
        "relevance": 0.1
    },
    {
        "cpe": """{
            "part": "a",
            "vendor": "microsoft",
            "product": "windows_10",
            "version": "20h2",
            "update": "",
            "edition": "",
            "language": ""
        }""",
        "relevance": 0.2
    },
    {
        "cpe": """{
            "part": "a",
            "vendor": "citrix",
            "product": "xenapp",
            "version": "7.15",
            "update": "",
            "edition": "ltsr",
            "language": ""
        }""",
        "relevance": 0.8
    },
    {
        "cpe": """{
            "part": "a",
            "vendor": "citrix",
            "product": "virtual_apps_and_desktops",
            "version": "1912",
            "update": "",
            "edition": "ltsr",
            "language": ""
        }""",
        "relevance": 0.9
    },
    {
        "cpe": """{
            "part": "h",
            "vendor": "dell",
            "product": "idrac",
            "version": "7.0",
            "update": "",
            "edition": "",
            "language": ""
        }""",
        "relevance": 0.5
    },
    {
        "cpe": """{
            "part": "a",
            "vendor": "citrix",
            "product": "workspace_app",
            "version": "2112",
            "update": "",
            "edition": "",
            "language": ""
        }""",
        "relevance": 0.7
    },
    {
        "cpe": """{
            "part": "a",
            "vendor": "mozilla",
            "product": "firefox",
            "version": "89.0",
            "update": "",
            "edition": "",
            "language": ""
        }""",
        "relevance": 0.4
    },
    {
        "cpe": """{
            "part": "a",
            "vendor": "google",
            "product": "chrome",
            "version": "90.0",
            "update": "",
            "edition": "",
            "language": ""
        }""",
        "relevance": 0.3
    },
    {
        "cpe": """{
            "part": "a",
            "vendor": "citrix",
            "product": "hdx_wmi_provider",
            "version": "1.0.0",
            "update": "",
            "edition": "",
            "language": ""
        }""",
        "relevance": 0.6
    },
    {
        "cpe": """{
            "part": "a",
            "vendor": "citrix",
            "product": "ica_client",
            "version": "7.0",
            "update": "",
            "edition": "",
            "language": ""
        }""",
        "relevance": 0.7
    },
    {
        "cpe": """{
            "part": "a",
            "vendor": "adobe",
            "product": "acrobat_reader",
            "version": "2021.001.20145",
            "update": "",
            "edition": "",
            "language": ""
        }""",
        "relevance": 0.3
    },
    {
        "cpe": """{
            "part": "h",
            "vendor": "citrix",
            "product": "netscaler_application_delivery_controller",
            "version": "-",
            "update": "",
            "edition": "enterprise",
            "language": ""
        }""",
        "relevance": 0.5
    }
]
# 将原始 20 个和新增 30 个文档合并
documents_cpe.extend(additional_documents_cpe)

# 构建成对的文本
sentence_pairs = [[query_cpe, doc["cpe"]] for doc in documents_cpe]

# 计算每对文本的得分
scores = model.compute_score(
    sentence_pairs,
    max_passage_length=128,
    weights_for_different_modes=[0.4, 0.2, 0.4]
)

# 打印结果
# 构建结果数据
results = []
print("查询与文档的得分:")
for i, (doc, colbert, sparse, dense, combined) in enumerate(
    zip(
        documents_cpe,
        scores['colbert'],
        scores['sparse'],
        scores['dense'],
        scores['colbert+sparse+dense']
    )
):
    print(f"文档 {i+1}: {doc['cpe']}")
    print(f"  ColBERT 得分: {colbert:.4f}")
    print(f"  稀疏得分: {sparse:.4f}")
    print(f"  稠密得分: {dense:.4f}")
    print(f"  综合得分 (ColBERT + 稀疏 + 稠密): {combined:.4f}")
    print(f"  标注相关性: {doc['relevance']:.1f}")
    print("\n")
    results.append({
        "Document Index": i + 1,
        "CPE": doc['cpe'],
        "ColBERT Score": colbert,
        "Sparse Score": sparse,
        "Dense Score": dense,
        "Combined Score": combined,
        "Relevance": doc['relevance']
    })

# 转换为 DataFrame
results_df = pd.DataFrame(results)
print(results_df.head(100))