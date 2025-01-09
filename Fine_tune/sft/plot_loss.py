#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Introduce :
@File      : plot_loss.py
@Time      : 2024/3/15 15:53
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
import json
import matplotlib.pyplot as plt

# Load the JSON data
# checkpoint_dir=r"/data/xw/alpaca-lora/train_llm/finetune_vulllm/output_vulllm_v4/checkpoint-20700"
# checkpoint_dir=r"/data/xw/alpaca-lora/train_llm/finetune_mm_llm/output_mmlllm_v5/"
# checkpoint_dir=r"/root/data/wjy/vip_vul_pro/Fine_tune/translate/checkpoints/checkpoint-416"
#checkpoint_dir=r"/root/data/wjy/vip_vul_pro/Fine_tune/translate/new_checkpoints/checkpoint-46"
checkpoint_dir=r"/root/data/wjy/vip_vul_pro/Fine_tune/translate/new_checkpoints/checkpoint-109"

# checkpoint_dir=r"/data/xw/alpaca-lora/train_llm/finetune_mm_llm/output_mmlllm_v8/checkpoint-49600"
# checkpoint_dir=r"/data/xw/alpaca-lora/train_llm/finetune_vulllm/output_vulllm_v5/checkpoint-1150"
# checkpoint_dir=r"/data/xw/alpaca-lora/train_llm/finetune_vulllm/output_vulllm_v5/checkpoint-1300"
# checkpoint_dir=r"/data/xw/alpaca-lora/train_llm/finetune_vulllm/output_vulllm_v5"
# checkpoint_dir=r"/data/xw/alpaca-lora/train_llm/finetune_vulllm/output_vulllm_v6/checkpoint-1740"
# checkpoint_dir=r"/data/xw/Qwen2/examples/sft/output_qwen2/checkpoint-810"
# checkpoint_dir=r"/data/xw/Qwen2/examples/sft/output_qwen2/checkpoint-2370"
with open(f'{checkpoint_dir}/trainer_state.json') as file:
    data = json.load(file)

# Extract steps, loss, and eval_loss
steps = []
loss = []
eval_steps=[]
eval_loss = []
flag=False
interval=5
for entry in data['log_history']:
    if 'step' in entry and 'loss' in entry:
        step= entry['step']
        if flag:
            if step % interval == 0:  # Check if step is a multiple of 100
                steps.append(step)
                loss.append(entry['loss'])  # Add corresponding loss
        else:
            steps.append(step)
            loss.append(entry['loss'])  # Add corresponding loss
    if 'eval_loss' in entry:
        eval_steps.append(entry['step'])
        eval_loss.append(entry['eval_loss'])

# Plot step vs loss
plt.figure(figsize=(10, 5))
plt.plot(steps, loss, label='Loss')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Step vs Loss')
plt.legend()
plt.grid(True)
# plt.show()

# Plot step vs eval_loss
if eval_loss:
    plt.figure(figsize=(10, 5))
    plt.plot(eval_steps, eval_loss, label='Eval Loss', color='red')
    plt.xlabel('Step')
    plt.ylabel('Eval Loss')
    plt.title('Step vs Eval Loss')
    plt.legend()
    plt.grid(True)
    # plt.show()
else:
    print("No eval_loss data available.")
plt.show()


import numpy as np
import matplotlib.pyplot as plt

# 移动平均函数
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


# 当 window_size 等于 1 时，移动平均实际上就是原始数据，不会进行任何平滑。
# 当 window_size 增加时，考虑的数据点变多，平滑程度增加，曲线变得更光滑。
# 当 window_size 很大时，曲线会变得非常平滑，但也可能丢失一些重要的波动信息。
# 使用移动平均平滑曲线
# window_size = 1  # 窗口大小，可以根据需要调整
# smoothed_loss = moving_average(loss, window_size)
# smoothed_steps = steps[window_size - 1:]  # 调整步数以匹配平滑后的损失大小
#
# # 绘制平滑后的曲线
# plt.figure(figsize=(10, 5))
# plt.plot(smoothed_steps, smoothed_loss, label='Smoothed Loss')
# plt.xlabel('Step')
# plt.ylabel('Loss')
# plt.title('Step vs Loss (Smoothed)')
# plt.legend()
# plt.grid(True)
# plt.show()
