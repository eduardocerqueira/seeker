#date: 2025-06-05T16:45:13Z
#url: https://api.github.com/gists/74195b153ed73f15cd01ab09b38b3e23
#owner: https://api.github.com/users/readyName

#!/bin/bash

# WAI Protocol Worker Node 部署脚本（macOS 版）
# 功能：自动化安装依赖、配置 WAI CLI、运行多实例 Worker、写入并加载环境变量
# 支持系统：macOS（M1、M2、M3、M4）
# 作者：基于 WAI Protocol 指南生成，优化为 macOS，包含环境变量处理

# 颜色定义，便于交互界面
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # 无颜色

# 日志函数
log() {
    echo -e "${GREEN}[INFO] $1${NC}"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}"
    exit 1
}

warn() {
    echo -e "${YELLOW}[WARN] $1${NC}"
}

# 检查 macOS 和 M 系列芯片
check_system() {
    log "正在检查系统环境..."
    if [[ "$(uname)" != "Darwin" ]]; then
        error "此脚本仅支持 macOS 系统，当前系统为 $(uname)"
    fi

    chip=$(sysctl -n machdep.cpu.brand_string)
    if [[ ! "$chip" =~ "Apple M" ]]; then
        warn "未检测到 Apple M 系列芯片，当前芯片为 $chip，可能不完全兼容"
        read -p "是否继续运行脚本？(y/n): " continue
        if [[ "$continue" != "y" && "$continue" != "Y" ]]; then
            error "用户选择退出"
        fi
    else
        log "检测到 Apple M 系列芯片：$chip"
    fi
}

# 检查和安装 Homebrew
install_homebrew() {
    log "检查 Homebrew 是否已安装..."
    if ! command -v brew &> /dev/null; then
        log "未检测到 Homebrew，正在安装..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        if [[ $? -ne 0 ]]; then
            error "Homebrew 安装失败，请手动安装 Homebrew 后重试"
        fi
        # 配置 Homebrew 环境变量
        log "配置 Homebrew 环境变量..."
        echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zshrc
        eval "$(/opt/homebrew/bin/brew shellenv)"
        log "Homebrew 安装成功"
    else
        log "Homebrew 已安装，更新 Homebrew..."
        brew update
    fi
}

# 安装依赖
install_dependencies() {
    log "安装必要的依赖..."

    # 安装通用工具
    brew install nano curl git wget jq automake autoconf htop
    if [[ $? -ne 0 ]]; then
        error "依赖安装失败，请检查 Homebrew 和网络连接"
    fi

    # 安装 Python
    log "安装 Python..."
    brew install python@3.11
    if ! command -v python3 &> /dev/null; then
        error "Python 安装失败，请检查 Homebrew 和网络连接"
    fi
    log "Python 安装成功，版本：$(python3 --version)"

    # 安装 Node.js
    log "安装 Node.js..."
    brew install node@22
    if ! command -v node &> /dev/null; then
        error "Node.js 安装失败，请检查 Homebrew 和网络连接"
    fi
    log "Node.js 安装成功，版本：$(node -v)"

    # 安装 Yarn 和 PM2
    log "安装 Yarn 和 PM2..."
    npm install -g yarn pm2
    if ! command -v yarn &> /dev/null || ! command -v pm2 &> /dev/null; then
        error "Yarn 或 PM2 安装失败，请检查 npm 和网络连接"
    fi
    log "Yarn 版本：$(yarn -v)"
    log "PM2 安装成功"
}

# 安装 WAI CLI
install_wai_cli() {
    log "安装 WAI CLI..."
    curl -fsSL https://app.w.ai/install.sh | bash
    if [[ $? -ne 0 ]]; then
        error "WAI CLI 安装失败，请检查网络连接或稍后重试"
    fi
    if ! command -v wai &> /dev/null; then
        error "WAI CLI 未正确安装，请检查安装过程"
    fi
    log "WAI CLI 安装成功"
}

# 获取用户输入
get_user_input() {
    log "请提供 WAI API 密钥"
    read -p "输入您的 WAI API 密钥: " api_key
    if [[ -z "$api_key" ]]; then
        error "API 密钥不能为空"
    fi
    export W_AI_API_KEY="$api_key"

    log "请输入要运行的 Worker 实例数量（建议根据硬件性能选择，例如 2-4）"
    read -p "实例数量: " instance_count
    if ! [[ "$instance_count" =~ ^[0-9]+$ ]] || [[ "$instance_count" -lt 1 ]]; then
        error "实例数量必须为正整数"
    fi
}

# 写入环境变量到 .zshrc
write_env_to_zshrc() {
    log "写入 WAI API 密钥到 ~/.zshrc..."
    zshrc_file="$HOME/.zshrc"

    # 检查是否已存在 W_AI_API_KEY
    if grep -Fx "export W_AI_API_KEY=$W_AI_API_KEY" "$zshrc_file" > /dev/null; then
        log "环境变量 W_AI_API_KEY 已存在，跳过写入"
    else
        echo "export W_AI_API_KEY=$W_AI_API_KEY" >> "$zshrc_file"
        if [[ $? -ne 0 ]]; then
            error "写入 ~/.zshrc 失败，请检查文件权限"
        fi
        log "环境变量已写入 ~/.zshrc"
    fi

    # 加载 .zshrc
    log "加载 ~/.zshrc 以应用环境变量..."
    source "$zshrc_file"
    if [[ $? -ne 0 ]]; then
        warn "加载 ~/.zshrc 失败，您可能需要手动运行 'source ~/.zshrc'"
    else
        log "环境变量加载成功"
    fi
}

# 配置 PM2 运行多实例
configure_pm2() {
    log "配置 PM2 以运行 $instance_count 个 Worker 实例..."

    # 创建 PM2 配置文件
    cat > wai.config.js <<EOF
module.exports = {
  apps: [{
    name: 'wai-node',
    script: 'wai',
    args: 'run',
    instances: $instance_count,
    autorestart: true,
    watch: false,
    max_memory_restart: '1G',
    env: {
      NODE_ENV: 'production',
      W_AI_API_KEY: '$W_AI_API_KEY'
    }
  }]
};
EOF

    if [[ $? -ne 0 ]]; then
        error "创建 PM2 配置文件失败"
    fi
    log "PM2 配置文件创建成功：wai.config.js"
}

# 启动 Worker
start_workers() {
    log "启动 WAI Worker 实例..."
    pm2 start wai.config.js
    if [[ $? -ne 0 ]]; then
        error "启动 Worker 失败，请检查 PM2 配置或 WAI CLI"
    fi
    log "Worker 实例已启动，请稍等几分钟以完成初始化"
    sleep 10
    pm2 list
    log "您可以通过以下命令查看 Worker 日志：pm2 logs wai-node"
}

# 提供监控命令
show_monitor_commands() {
    log "以下是常用监控命令："
    echo "  - 查看 Worker 状态：pm2 list"
    echo "  - 查看 Worker 日志：pm2 logs wai-node"
    echo "  - 查看特定 Worker 日志：pm2 logs <ID>（例如 pm2 logs 0）"
    echo "  - 停止 Worker：pm2 stop wai.config.js"
    echo "  - 重启 Worker：pm2 restart wai.config.js"
    echo "  - 停止并删除所有 Worker：pm2 delete wai-node"
    echo "  - 查看 CPU 和内存使用：htop"
    echo "  - 查看磁盘使用：du -sh ~/.wombo"
}

# 主函数
main() {
    log "开始部署 WAI Protocol Worker Node（macOS 版）..."
    check_system
    install_homebrew
    install_dependencies
    install_wai_cli
    get_user_input
    write_env_to_zshrc
    configure_pm2
    start_workers
    show_monitor_commands
    log "部署完成！Worker 正在运行，请使用 'pm2 logs wai-node' 检查状态"
    log "如需调整实例数量，请修改 wai.config.js 文件并运行 'pm2 restart wai.config.js'"
}

# 执行主函数
main