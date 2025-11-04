#date: 2025-11-04T17:05:11Z
#url: https://api.github.com/gists/c3aaab267950da69f58133dc87ddea6a
#owner: https://api.github.com/users/xLmiler

#!/bin/bash

# 脚本出现错误时立即退出
set -e

# --- 配置 ---
REPO_URL="https://github.com/SillyTavern/SillyTavern"
BRANCH="staging"
DIR_NAME="SillyTavern"
# --- 结束配置 ---

# --- 辅助函数：彩色输出 ---
cecho() {
  local text="$1"
  local color="$2"
  
  case "$color" in
    red)    printf "\033[0;31m%s\033[0m\n" "$text" ;;
    green)  printf "\033[0;32m%s\033[0m\n" "$text" ;;
    yellow) printf "\033[0;33m%s\033[0m\n" "$text" ;;
    blue)   printf "\033[0;34m%s\033[0m\n" "$text" ;;
    *)      echo "$text" ;;
  esac
}

# --- 辅助函数：检查命令是否存在并提供安装指导 ---
check_dependency() {
  local cmd="$1"
  local package_name="$2"

  if ! command -v "$cmd" &> /dev/null; then
    cecho "错误: 未检测到 '$cmd' 命令。" "red"
    cecho "请先安装 '$package_name'。" "yellow"
    
    # 根据操作系统提供安装建议
    local os
    os=$(uname -s)
    case "$os" in
      Linux)
        if command -v apt-get &> /dev/null; then
          cecho "在 Debian/Ubuntu 上，可以尝试: sudo apt update && sudo apt install $package_name" "blue"
        elif command -v dnf &> /dev/null; then
          cecho "在 Fedora/CentOS 上，可以尝试: sudo dnf install $package_name" "blue"
        elif command -v pacman &> /dev/null; then
          cecho "在 Arch Linux 上，可以尝试: sudo pacman -S $package_name" "blue"
        fi
        ;;
      Darwin) # macOS
        if command -v brew &> /dev/null; then
          cecho "在 macOS 上，可以尝试: brew install $package_name" "blue"
        fi
        ;;
    esac
    exit 1
  fi
}

# --- 主逻辑 ---
main() {
  cecho "--- SillyTavern 启动脚本 ---" "blue"

  # 1. 检查依赖：Git 和 Node.js
  cecho "[1/4] 正在检查依赖..."
  check_dependency "git" "git"
  check_dependency "node" "nodejs"
  cecho "所有依赖均已安装。" "green"

  # 2. 处理 SillyTavern 目录
  if [ ! -d "$DIR_NAME" ]; then
    cecho "[2/4] '$DIR_NAME' 目录不存在，开始克隆仓库..."
    git clone --depth 1 -b "$BRANCH" "$REPO_URL" "$DIR_NAME"
    cecho "仓库克隆完成。" "green"
  else
    cecho "[2/4] '$DIR_NAME' 目录已存在，尝试更新..."
    cd "$DIR_NAME"
    # 检查是否为Git仓库
    if [ -d ".git" ]; then
      # 保存当前分支，拉取后切换回来
      current_branch=$(git rev-parse --abbrev-ref HEAD)
      git fetch origin
      git reset --hard "origin/$BRANCH" # 强制更新到远程分支的最新状态
      # git pull origin "$BRANCH" # 或者使用更安全的pull
      cecho "仓库更新完成。" "green"
    else
      cecho "'$DIR_NAME' 目录存在但不是一个Git仓库，跳过更新。" "yellow"
    fi
    cd .. # 返回上一级目录
  fi
  
  # 3. 进入目录并安装Node模块
  cecho "[3/4] 正在进入 '$DIR_NAME' 目录并安装/更新依赖模块..."
  cd "$DIR_NAME"
  export NODE_ENV=production
  # --prefer-offline 可以在有缓存的情况下加速安装
  npm install --no-audit --no-fund --loglevel=error --no-progress --omit=dev --prefer-offline
  cecho "Node模块安装完成。" "green"
  
  # 4. 启动服务器
  cecho "[4/4] 正在启动 SillyTavern 服务器..." "blue"
  cecho "----------------------------------------"
  # 使用 "$@" 将所有传递给脚本的参数原封不动地传递给 server.js
  node server.js "$@"
}

# --- 执行主函数 ---
main "$@"