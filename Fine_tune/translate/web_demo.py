
from argparse import ArgumentParser
from threading import Thread

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

DEFAULT_CKPT_PATH = "/data/huggingface_models/safe_v3_checkpoint60" #更改模型路径



def _get_args():
    """
        解析命令行参数，用于启动聊天机器人程序。
        客户可以通过这些参数自定义运行的配置，如模型路径、是否使用CPU等。
        """
    parser = ArgumentParser(description="Appsec bot web chat demo.")
    parser.add_argument(
        "-c",
        "--checkpoint-path",
        type=str,
        default=DEFAULT_CKPT_PATH,
        help="Checkpoint name or path, default to %(default)r",
    )
    parser.add_argument(
        "--cpu-only", action="store_true", help="Run demo with CPU only"
    )

    parser.add_argument(
        "--share",
        action="store_true",
        default=False,
        help="Create a publicly shareable link for the interface.",
    )
    parser.add_argument(
        "--inbrowser",
        action="store_true",
        default=False,
        help="Automatically launch the interface in a new tab on the default browser.",
    )
    parser.add_argument(
        "--server-port", type=int, default=58001, help="Demo server port."
    )
    parser.add_argument(
        "--server-name", type=str, default="0.0.0.0", help="Demo server name."
    )

    args = parser.parse_args()
    return args


def _load_model_tokenizer(args):
    """
       加载预训练模型和分词器，用于支持自然语言理解和生成。
       根据用户的设备配置，支持使用CPU或GPU运行。
       """
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint_path,
        resume_download=True,
    )

    if args.cpu_only:
        device_map = "cpu"
    else:
        device_map = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_path,
        torch_dtype="auto",
        device_map=device_map,
        resume_download=True,
    ).eval()
    model.generation_config.max_new_tokens = 2048  # For chat.

    return model, tokenizer

# 聊天功能实现（流式生成）
def _chat_stream(model, tokenizer, query, history):
    """
        流式生成聊天机器人的回答。
        用户的输入和历史记录会被拼接后送入模型，模型逐步返回生成的回答。
        """
    conversation = []
    for query_h, response_h in history:
        conversation.append({"role": "user", "content": query_h})
        conversation.append({"role": "assistant", "content": response_h})
    conversation.append({"role": "user", "content": query}) # 添加当前用户的提问
    input_text = tokenizer.apply_chat_template(
        conversation,
        add_generation_prompt=True, # 自动添加生成提示
        tokenize=False, # 不提前分词
    )
    # 将文本转换为模型输入
    inputs = tokenizer([input_text], return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(
        tokenizer=tokenizer, skip_prompt=True, timeout=60.0, skip_special_tokens=True
    )
    # 设置生成配置
    generation_kwargs = {
        **inputs,
        "streamer": streamer,
    }
    # 通过线程运行生成，防止界面卡顿
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    # 实时返回生成的文本
    for new_text in streamer:
        yield new_text


def _gc():
    import gc
    """
       清理内存和显存，以防止显存溢出或内存占用过高。
       """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _launch_demo(args, model, tokenizer):
    """
        使用Gradio创建聊天机器人Web界面，并根据用户操作处理聊天逻辑。
        """
    def predict(_query, _chatbot, _task_history):
        """
                处理用户的提问并返回机器人生成的回答。
                支持流式输出，回答会逐步显示在界面上。
                """
        print(f"User: {_query}")
        _chatbot.append((_query, ""))
        full_response = ""
        response = ""
        for new_text in _chat_stream(model, tokenizer, _query, history=_task_history):
            response += new_text
            _chatbot[-1] = (_query, response)

            yield _chatbot
            full_response = response

        print(f"History: {_task_history}")
        _task_history.append((_query, full_response))
        print(f"Appsec bot: {full_response}")

    def regenerate(_chatbot, _task_history):
        if not _task_history:
            yield _chatbot
            return
        item = _task_history.pop(-1)
        _chatbot.pop(-1)
        yield from predict(item[0], _chatbot, _task_history)

    def reset_user_input():
        return gr.update(value="")

    def reset_state(_chatbot, _task_history):
        _task_history.clear()
        _chatbot.clear()
        _gc()
        return _chatbot

    with gr.Blocks(css="footer{display:none !important}") as demo:
        gr.Markdown(
            """\
<center><font size=3>This WebUI is based on Appsec bot, developed by 软评中心. \
(本WebUI基于Appsec bot打造，实现聊天机器人功能。)</center>"""
        )

        chatbot = gr.Chatbot(label="Appsec bot", elem_classes="control-height")
        query = gr.Textbox(lines=2, label="Input")
        task_history = gr.State([])

        with gr.Row():
            empty_btn = gr.Button("🧹 Clear History (清除历史)")
            submit_btn = gr.Button("🚀 Submit (发送)")
            regen_btn = gr.Button("🤔️ Regenerate (重试)")

        submit_btn.click(
            predict, [query, chatbot, task_history], [chatbot], show_progress=True
        )
        submit_btn.click(reset_user_input, [], [query])
        empty_btn.click(
            reset_state, [chatbot, task_history], outputs=[chatbot], show_progress=True
        )
        regen_btn.click(
            regenerate, [chatbot, task_history], [chatbot], show_progress=True
        )

        gr.Markdown("""\
<font size=2>Note: This demo is governed by the original license of Appsec bot. \
We strongly advise users not to knowingly generate or allow others to knowingly generate harmful content, \
including hate speech, violence, pornography, deception, etc. \
(注：本演示受Appsec bot的许可协议限制。我们强烈建议，用户不应传播及不应允许他人传播以下内容，\
包括但不限于仇恨言论、暴力、色情、欺诈相关的有害信息。)""")

    demo.queue().launch(
        share=args.share,
        inbrowser=args.inbrowser,
        server_port=args.server_port,
        server_name=args.server_name,
        # show_api=False,
        quiet=True,  # 隐藏启动信息
        # show_tips=False,  # 隐藏提示信息
        # show_api=False,  # 隐藏 API 标志
    )


def main():
    """
        主函数：解析参数、加载模型与分词器，并启动聊天机器人界面。
        """
    args = _get_args()# 获取命令行参数

    model, tokenizer = _load_model_tokenizer(args) # 加载模型和分词器

    _launch_demo(args, model, tokenizer) # 启动聊天界面


if __name__ == "__main__":
    main()

