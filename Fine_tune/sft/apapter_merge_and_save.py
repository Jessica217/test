#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Introduce :
@File      : apapter_merge_and_save.py
@Time      : 2023/12/29 14:56
@Author    : wei.xu
@Tel       : 
@Email     : wei.xu@tophant.com
@pip       : pip install 
"""
# import crsts
import sys
import os
import time
import datetime
import numpy as np
import pandas as pd
import warnings
import torch
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
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer


# path_to_adapter=r"/data/xw/alpaca-lora/train_llm/finetune_vulllm/output_vulllm_v5/checkpoint-1300"
# path_to_adapter=r"/data/xw/alpaca-lora/train_llm/finetune_vulllm/output_vulllm_v6/checkpoint-1440"
# path_to_adapter=r"/root/data/wjy/vip_vul_pro/Fine_tune/replace/dataset_user_info/output_augdata/checkpoint-40"
path_to_adapter=r"/root/data/wjy/vip_vul_pro/Fine_tune/replace/dataset_user_info/output_converted_data/checkpoint-155"


# path_to_adapter=r"/data/checkpoint-1100"
new_model_directory=r"/data/huggingface_models/appsec_bot-v4"
model = AutoPeftModelForCausalLM.from_pretrained(
    path_to_adapter, # path to the output directory
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
).eval()

merged_model = model.merge_and_unload()
# max_shard_size and safe serialization are not necessary.
# They respectively work for sharding checkpoint and save the model to safetensors
merged_model.save_pretrained(new_model_directory, max_shard_size="2048MB", safe_serialization=True)

tokenizer = AutoTokenizer.from_pretrained(
    path_to_adapter, # path to the output directory
    trust_remote_code=True
)
tokenizer.save_pretrained(new_model_directory)
print("end of saved model")
