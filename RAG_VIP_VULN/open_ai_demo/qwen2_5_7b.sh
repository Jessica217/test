#!/bin/bash

# 设置重试次数
max_retries=100

# 设置当前重试次数
retry_count=0

# 定义您想运行的命令
#command="./hfd.sh Qwen/Qwen2.5-7B-Instruct --tool aria2c -x 8 --hf_username jessicayaya  --hf_token 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"
#/root/data/huggingface_models/hfd.sh
command="/root/data/huggingface_models/hfd.sh
 Qwen/Qwen2.5-72B-Instruct --tool aria2c -x 8 --hf_username jessicayaya  --hf_token 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"


# 定义日志文件
log_file="t1111.log"

# 循环尝试运行命令，直到成功或达到最大重试次数
while [ $retry_count -lt $max_retries ]; do
  # 运行命令并重定向输出到日志文件
  nohup $command > $log_file 2>&1 &

  # 等待命令执行完成
  wait $!

  # 检查命令是否成功
  if [ $? -eq 0 ]; then
    echo "命令成功执行！"
    exit 0
  else
    echo "命令执行失败，重试中... (第 $((retry_count+1)) 次)"
    retry_count=$((retry_count+1))
  fi
done

echo "命令在 $max_retries 次尝试后仍然失败。"
exit 1
