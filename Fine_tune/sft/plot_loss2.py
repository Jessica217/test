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

# checkpoint_dir=r"/data/xw/alpaca-lora/train_llm/finetune_mm_llm/output_mmlllm_v8/checkpoint-49600"
# checkpoint_dir=r"/data/xw/alpaca-lora/train_llm/finetune_vulllm/output_vulllm_v5/checkpoint-1100"
# checkpoint_dir=r"/data/xw/alpaca-lora/train_llm/finetune_vulllm/output_vulllm_v6/checkpoint-1200"
# checkpoint_dir=r"/data/xw/alpaca-lora/train_llm/finetune_vulllm/output_vulllm_v6/checkpoint-1740"
# checkpoint_dir=r"/data/xw/Qwen2/examples/sft/output_qwen2/checkpoint-810"
checkpoint_dir=r"/root/data/wjy/vip_vul_pro/Fine_tune/translate/checkpoints/checkpoint-416"

with open(f'{checkpoint_dir}/trainer_state.json') as file:
    data = json.load(file)

# Extract steps, loss, and eval_loss
steps = []
loss = []
eval_steps = []
eval_loss = []
flag=True
interval=5
for entry in data['log_history']:
    if 'step' in entry and 'loss' in entry:
        step= entry['step']
        if 300 <= step <= 400:  # Between step 300 and 400, plot every 20 steps
            if step % 20 == 0:
                steps.append(step)
                loss.append(entry['loss'])  # Add corresponding loss
        else:
            if flag:
                if step % interval == 0:  # For other steps, plot every 5 steps
                    steps.append(step)
                    loss.append(entry['loss'])  # Add corresponding loss
            else:
                steps.append(step)
                loss.append(entry['loss'])  # Add corresponding loss
    if 'eval_loss' in entry:
        eval_steps.append(entry['step'])
        eval_loss.append(entry['eval_loss'])

# Plot step vs loss and eval_loss
plt.figure(figsize=(10, 5))
plt.plot(steps, loss, label='Loss')
if eval_loss:
    plt.plot(eval_steps, eval_loss, label='Eval Loss', color='red')

# Set specific tick positions for the x-axis (steps)
xticks = np.arange(0, 401, 20)  # Display steps from 320 to 400 with a step of 20
plt.xticks(xticks)

plt.xlabel('Step')
plt.ylabel('Loss / Eval Loss')
plt.title('Step vs Loss / Eval Loss')
plt.legend()
plt.grid(True)
plt.show()
