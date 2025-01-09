# This code is based on the revised code from fastchat based on tatsu-lab/stanford_alpaca.


import json
import logging
import os
import pathlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from safetensors.torch import load_file

import torch
import transformers
from accelerate.utils import DistributedType
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
# from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training,
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training,set_peft_model_state_dict

from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    #deepspeed,
)
from transformers.trainer_pt_utils import LabelSmoother

from pathlib import Path


IGNORE_TOKEN_ID = LabelSmoother.ignore_index

TEMPLATE = "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content']}}{% if loop.last %}{{ '<|im_end|>'}}{% else %}{{ '<|im_end|>\n' }}{% endif %}{% endfor %}"

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


@dataclass
class ModelArguments:
    # model_name_or_path: Optional[str] = field(default="Qwen/Qwen2-7B")
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2-14B")


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    lazy_preprocess: bool = False


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=8192,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    use_lora: bool = False


@dataclass
class LoraArguments:
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "gate_proj",
            "down_proj",
        ]
    )
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False


def maybe_zero_3(param):
    if hasattr(param, "ds_id"):
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return


def safe_save_model_for_hf_trainer(
    trainer: transformers.Trainer, output_dir: str, bias="none"
):
    """Collects the state dict and dump to disk."""
    # check if zero3 mode enabled
    if transformers.integrations.is_deepspeed_zero3_enabled():
        state_dict = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
    else:
        if trainer.args.use_lora:
            state_dict = get_peft_state_maybe_zero_3(
                trainer.model.named_parameters(), bias
            )
        else:
            state_dict = trainer.model.state_dict()
    if trainer.args.should_save and trainer.args.local_rank == 0:
        trainer._save(output_dir, state_dict=state_dict)


def preprocess(
    messages,
    tokenizer: transformers.PreTrainedTokenizer,
    max_len: int,
) -> Dict:
    """Preprocesses the data for supervised fine-tuning."""

    texts = []
    for i, msg in enumerate(messages):
        texts.append(
            tokenizer.apply_chat_template(
                msg,
                chat_template=TEMPLATE,
                tokenize=True,
                add_generation_prompt=False,
                padding="max_length",
                max_length=max_len,
                truncation=True,
            )
        )
    input_ids = torch.tensor(texts, dtype=torch.int)
    #print(f"inputs['input_ids']:{inputs['input_ids']}")
    target_ids = input_ids.clone()
    target_ids[target_ids == tokenizer.pad_token_id] = IGNORE_TOKEN_ID
    attention_mask = input_ids.ne(tokenizer.pad_token_id)

    return dict(
        input_ids=input_ids, target_ids=target_ids, attention_mask=attention_mask
    )


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self, raw_data, tokenizer: transformers.PreTrainedTokenizer, max_len: int
    ):
        super(SupervisedDataset, self).__init__()

        rank0_print("Formatting inputs...")
        # print('type(raw_data)', type(raw_data))
        # print('raw_data',len(raw_data))
        # for example in raw_data:
        #     messages.append(example["messages"])

        messages = [example["messages"] for example in raw_data]
        data_dict = preprocess(messages, tokenizer, max_len)

        self.input_ids = data_dict["input_ids"]
        self.target_ids = data_dict["target_ids"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.target_ids[i],
            attention_mask=self.attention_mask[i],
        )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self, raw_data, tokenizer: transformers.PreTrainedTokenizer, max_len: int
    ):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]
        print(f"raw_data[i]: {self.raw_data[i]}")
        ret = preprocess([self.raw_data[i]["messages"]], self.tokenizer, self.max_len)
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["target_ids"][0],
            attention_mask=ret["attention_mask"][0],
        )
        self.cached_data_dict[i] = ret

        return ret


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args,
    max_len,
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = (
        LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
    )
    rank0_print("Loading data...")

    train_data = []
    with open(data_args.data_path, "r") as f: #数据格式为json格式 而非jsonl json格式可直接load，不需要遍历
        train_data = json.load(f)

    import random
    print(f"shuffling {len(train_data)} train data to avoid overfitting...")
    random.shuffle(train_data)
    per_device_train_batch_size=training_args.per_device_train_batch_size
    gradient_accumulation_steps=training_args.gradient_accumulation_steps
    print(f"per_device_train_batch_size: {per_device_train_batch_size}")
    print(f"gradient_accumulation_steps: {gradient_accumulation_steps}")
    max_steps= len(train_data)/(per_device_train_batch_size*gradient_accumulation_steps)
    print(f"max_steps: {max_steps}")
    train_dataset = dataset_cls(train_data, tokenizer=tokenizer, max_len=max_len)

    if data_args.eval_data_path:
        # eval_data = []
        with open(data_args.eval_data_path, "r") as f:
            eval_data = json.load(f)
        eval_dataset = dataset_cls(eval_data, tokenizer=tokenizer, max_len=max_len)
    else:
        eval_dataset = None

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)

def load_checkpoint(model, resume_from_checkpoint):
    """
    如果可用，使用 pathlib 进行路径处理，从检查点加载模型权重。

    Args:
    model (torch.nn.Module): 要加载权重的模型。
    resume_from_checkpoint (str): 检查点所在的目录路径。

    Returns:
    torch.nn.Module: 加载了权重的模型。
    """
    if resume_from_checkpoint:
        # 使用 pathlib 创建检查点目录的路径
        checkpoint_dir = Path(resume_from_checkpoint)
        # 定义完整模型检查点的路径
        full_checkpoint_path = checkpoint_dir / "pytorch_model.bin"
        # 定义 LoRA 模型检查点的路径
        lora_checkpoint_path = checkpoint_dir / "adapter_model.safetensors"

        print(f'checkpoint_dir是否存在:{checkpoint_dir.exists()}')

        # 如果完整模型检查点存在
        if full_checkpoint_path.exists():
            checkpoint_path = full_checkpoint_path
            print(f"从完整模型检查点重启: {checkpoint_path}")
            # 加载检查点并更新模型权重
            adapters_weights = torch.load(checkpoint_path)
            model.load_state_dict(adapters_weights)
        # 如果 LoRA 模型检查点存在
        elif lora_checkpoint_path.exists():
            checkpoint_path = lora_checkpoint_path
            print(f"从 LoRA 模型检查点重启: {checkpoint_path}")
            # 加载检查点并使用特定函数更新模型状态
            #adapters_weights = torch.load(checkpoint_path)
            adapters_weights = load_file(checkpoint_path)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            # 如果没有找到检查点
            print(f"在 {checkpoint_dir} 中未找到检查点")
            return model

    return model

def train():
    global local_rank
    global training_args
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        lora_args,
    ) = parser.parse_args_into_dataclasses()
    print(f"****************************Model Args*****************************")
    print(f"model args:\n{model_args}")
    print(f"*****************************Data Args*****************************")
    print(f"data args:\n{data_args}")
    print(f"**************************Lora Args ******************************")
    print(f"lora args:\n{lora_args}")
    print(f"**************************Training Arguments ***********************")
    print(f"training args:\n{training_args}")
    # This serves for single-gpu qlora.
    if (
        getattr(training_args, "deepspeed", None)
        and int(os.environ.get("WORLD_SIZE", 1)) == 1
    ):
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    local_rank = training_args.local_rank
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1

    # device_map = None
    # device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else "auto"
    # if lora_args.q_lora:
    #     device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else "auto"
    #     if len(training_args.fsdp) > 0 or deepspeed.is_deepspeed_zero3_enabled():
    #         logging.warning("FSDP or ZeRO3 is incompatible with QLoRA.")

    device_map = 'balanced_low_0'

    print(f"device_map:{device_map}")
    print(f"world_size:{world_size}")
    print(f"ddp:{ddp}")
    model_load_kwargs = {
        "low_cpu_mem_usage": not transformers.integrations.is_deepspeed_zero3_enabled(),
    }
    low_cpu_mem_usage=True
    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )
    print(f"compute_dtype:{compute_dtype}")

    # Load model and tokenizer
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    config.use_cache = False
    print(f"config:\n{config}")
    print(f"device_map:{device_map}")
    print(f"***********************Loading Model*************************")
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        torch_dtype=compute_dtype,
        device_map=device_map,
        resume_download=True
    )

    # print(f"model:\n{model}")
    print(f"******************************Loading Tokenizer*********************")
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
        resume_download=True,
    )

    if training_args.use_lora:
        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=lora_args.lora_target_modules,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if lora_args.q_lora:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )

        model = get_peft_model(model, lora_config)
        resume_from_checkpoint = training_args.resume_from_checkpoint
        print(f"resume_from_checkpoint :{resume_from_checkpoint}")
        model = load_checkpoint(model, resume_from_checkpoint)
        print(f"*************************Peft Model******************")
        print(f"peft model: \n {model}")
        # Print peft trainable params
        print(f"***************************Print peft trainable params***********************")
        model.print_trainable_parameters()

        if training_args.gradient_checkpointing:
            model.enable_input_require_grads()

    # Load data
    print(f"************************************Loading data*********************************")
    import datetime
    start = datetime.datetime.now()
    data_module = make_supervised_data_module(
        tokenizer=tokenizer, data_args=data_args, max_len=training_args.model_max_length
    )
    end = datetime.datetime.now()
    print(f"Dataset loaded in {(end - start).total_seconds()} s")

    # Start trainer
    print("********************************Start Trainning******************************")
    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )

    # `not training_args.use_lora` is a temporary workaround for the issue that there are problems with
    # loading the checkpoint when using LoRA with DeepSpeed.
    # Check this issue https://github.com/huggingface/peft/issues/746 for more information.
    if (
        list(pathlib.Path(training_args.output_dir).glob("checkpoint-*"))
        and not training_args.use_lora
    ):
        trainer.train(resume_from_checkpoint=True)
    else:
        #trainer.train()
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_state()

    safe_save_model_for_hf_trainer(
        trainer=trainer, output_dir=training_args.output_dir, bias=lora_args.lora_bias
    )
    print(f"finished training")
    print(f"checkpoint saved to {training_args.output_dir}")


if __name__ == "__main__":
    # 将 'max_split_size_mb' 注意这里的单位是 MB，所以需要将 GB 转换为 MB
    os.environ["PYTORCH_CUDA_ALLOC_CONF_ALLOC_CONF"] = "max_split_size_mb:1024"  # prevents the native allocator from splitting blocks larger than this size (in MB).
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'garbage_collection_threshold:0.85'  # 设置此阈值（例如，0.8）后，如果GPU内存容量使用超过阈值（即，GPU应用程序分配的总内存的80%），分配器将开始回收GPU内存块

    train()
