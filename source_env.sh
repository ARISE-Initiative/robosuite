#!/bin/bash

# 获取当前脚本的绝对路径
current_directory=$(cd "$(dirname "$0")"; pwd)

# 将当前文件夹路径添加到PYTHONPATH
export PYTHONPATH="$PYTHONPATH:$current_directory"

echo "Current directory added to PYTHONPATH: $current_directory"