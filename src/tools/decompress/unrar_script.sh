#!/bin/bash

# 脚本会直接在运行的目录下进行相关的操作：
# --- 脚本配置 ---
LOG_FILE="unrar_script.log"  # 日志文件路径
DATE=$(date "+%Y-%m-%d %H:%M:%S")

# --- 日志函数 (仅日志文件输出) ---
log_info() {
  echo "[$DATE] - INFO - $1" >> "$LOG_FILE"
  # echo "[$DATE] - INFO - $1"  # 注释掉控制台输出
}

log_error() {
  echo "[$DATE] - ERROR - $1" >> "$LOG_FILE"
  # echo "[$DATE] - ERROR - $1"  # 注释掉控制台输出
}

# --- 检查 unrar 是否安装 ---
if ! command -v unrar &> /dev/null
then
  log_error "错误: unrar 命令未找到。请先安装 unrar (例如: sudo apt install unrar 或 sudo yum install unrar)。"
  exit 1
fi

# --- 查找并处理 RAR 文件 ---
find . -type f -name "*.rar" -print0 | while IFS= read -r -d $'\0' rar_file; do
  log_info "开始处理 RAR 文件: \"$rar_file\""

  # 获取 RAR 文件所在目录
  rar_dir=$(dirname "$rar_file")

  # 解压 RAR 文件到当前目录，**保留文件名和目录结构**
  log_info "解压 \"$rar_file\" 到目录: \"$rar_dir\" (保留文件名和目录结构)"
  unrar x -o+ "$rar_file" "$rar_dir" > /dev/null 2>&1  # 使用 'x' 命令保留路径, -o+ 覆盖已存在文件， > /dev/null 2>&1 静默输出

  if [ $? -eq 0 ]; then
    log_info "成功解压 \"$rar_file\""

    # 删除源 RAR 文件
    log_info "删除源 RAR 文件: \"$rar_file\""
    rm -f "$rar_file"

    if [ $? -eq 0 ]; then
      log_info "成功删除源文件 \"$rar_file\""
    else
      log_error "删除源文件 \"$rar_file\" 失败!"
    fi
  else
    log_error "解压 \"$rar_file\" 失败!"
  fi
  echo "" # 增加一个空行，分隔日志
done

log_info "RAR 文件处理完成。"

exit 0