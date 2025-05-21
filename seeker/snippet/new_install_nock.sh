#date: 2025-05-21T16:51:23Z
#url: https://api.github.com/gists/ebc1b38dd8fa8470c9407296e240d763
#owner: https://api.github.com/users/readyName

#!/bin/bash
set -e

# === 基本信息 ===
LOGFILE="$HOME/install_log_$(date +%F_%H-%M-%S).log"
REPO_URL="https://github.com/zorp-corp/nockchain"
PROJECT_DIR="nockchain"
WALLET_FILE="./target/wallet_keys.txt"
WALLET_CMD="./target/release/nockchain-wallet"
ENV_FILE=".env"

# === 日志记录 ===
exec > >(tee -a "$LOGFILE") 2>&1

# === 检测操作系统 & 核心数 ===
OS=""
CORES=1
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
  OS="linux"
  CORES=$(nproc)
  SYSTEM_MODEL=$(grep -m1 "model name" /proc/cpuinfo | awk -F: '{print $2}' | xargs)
elif [[ "$OSTYPE" == "darwin"* ]]; then
  OS="macos"
  CORES=$(sysctl -n hw.ncpu)
  SYSTEM_MODEL=$(system_profiler SPHardwareDataType | grep "Model Name" | awk -F: '{print $2}' | xargs)
else
  echo "❌ 不支持的系统类型：$OSTYPE"
  exit 1
fi

echo "✅ 检测到系统：$OS"
echo "✅ CPU 核心数：$CORES"
echo "✅ 系统型号：${SYSTEM_MODEL:-未知}"
sleep 2

echo -e "\n📦 [1/9] 安装依赖..."
if [ "$OS" == "macos" ]; then
  if ! command -v brew &>/dev/null; then
    echo "📥 安装 Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    if [[ -d "/opt/homebrew/bin" ]]; then
      echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
      eval "$(/opt/homebrew/bin/brew shellenv)"
    elif [[ -d "/usr/local/bin" ]]; then
      echo 'eval "$(/usr/local/bin/brew shellenv)"' >> ~/.bash_profile
      eval "$(/usr/local/bin/brew shellenv)"
    fi
  fi
  brew update
  brew install curl git wget lz4 jq make gcc automake autoconf tmux htop pkg-config openssl leveldb ncdu unzip libtool cmake screen || true
else
  sudo apt update
  sudo apt install -y curl git wget lz4 jq make gcc automake autoconf tmux htop pkg-config libssl-dev libleveldb-dev ncdu unzip libtool build-essential cmake screen clang llvm-dev libclang-dev || true
fi

echo -e "\n🦀 [2/9] 安装 Rust..."
if ! command -v rustup &>/dev/null; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
  source "$HOME/.cargo/env"
fi
rustup default stable
source "$HOME/.cargo/env"

echo -e "\n📁 [3/9] 拉取 nockchain 仓库..."
if [ -d "$PROJECT_DIR" ]; then
  echo "⚠️ 检测到已有 $PROJECT_DIR，是否删除？(y/n)"
  read -r confirm
  if [[ "$confirm" == "y" || "$confirm" == "Y" ]]; then
    rm -rf "$PROJECT_DIR"
    git clone "$REPO_URL"
  fi
else
  git clone "$REPO_URL"
fi
cd "$PROJECT_DIR"

# Step 4: 创建 .env 文件（防止 make 报错）
echo -e "\n📝 [4/9] 准备 .env 配置文件..."
if [ ! -f "$ENV_FILE" ]; then
  cp .env_example "$ENV_FILE"
  echo "✅ 已创建 .env 文件"
else
  echo "➡️ 已存在 .env 文件"
fi

# Step 5: 编译
echo -e "\n🛠️ [5/9] 开始构建..."
make install-hoonc
make build
make install-nockchain
make install-nockchain-wallet

# Step 6: 钱包处理
echo -e "\n🔐 [6/9] 钱包操作"
echo "1) 生成新钱包"
echo "2) 使用已有公钥"
read -rp "请选择 (1/2): " wallet_choice

if [[ "$wallet_choice" == "1" ]]; then
  if [ ! -x "$WALLET_CMD" ]; then
    echo "❌ 钱包程序不存在：$WALLET_CMD"
    exit 1
  fi

  wallet_output=$($WALLET_CMD keygen 2>&1 | tr -d '\0')
  echo "$wallet_output"

  pubkey=$(echo "$wallet_output" | grep -A1 "New Public Key" | tail -n1 | tr -d '"')
  privkey=$(echo "$wallet_output" | grep -A1 "New Private Key" | tail -n1 | tr -d '"')

  if [[ -n "$pubkey" && -n "$privkey" ]]; then
    mkdir -p "$(dirname "$WALLET_FILE")"
    echo -e "Public Key: $pubkey\nPrivate Key: $privkey" > "$WALLET_FILE"
    echo "💾 钱包密钥已保存：$WALLET_FILE"
  else
    echo "❌ 钱包生成失败，输出：$wallet_output"
    exit 1
  fi
elif [[ "$wallet_choice" == "2" ]]; then
  read -rp "请输入你的公钥: " pubkey
  if [[ -z "$pubkey" ]]; then
    echo "❌ 公钥不能为空"
    exit 1
  fi
else
  echo "❌ 无效选项，退出安装"
  exit 1
fi

# Step 7: 写入 .env
echo -e "\n🔧 [7/9] 写入 .env 配置..."
if grep -q "^MINING_PUBKEY=" "$ENV_FILE"; then
  sed -i.bak "s/^MINING_PUBKEY=.*/MINING_PUBKEY=$pubkey/" "$ENV_FILE"
else
  echo "MINING_PUBKEY=$pubkey" >> "$ENV_FILE"
fi
echo "✅ 公钥已写入 .env：$pubkey"

# Step 8: 环境变量配置
echo -e "\n🌍 [8/9] 写入环境变量..."
if [ -n "$ZSH_VERSION" ]; then
  PROFILE="$HOME/.zshrc"
elif [ -n "$BASH_VERSION" ]; then
  PROFILE="$HOME/.bashrc"
else
  PROFILE="$HOME/.profile"
fi

{
  echo ''
  echo '# === nockchain 配置 ==='
  echo "export PATH=\"\$PATH:$PWD/target/release\""
  echo "export RUST_LOG=info"
  echo "export MINIMAL_LOG_FORMAT=true"
} >> "$PROFILE"

export PATH="$PATH:$PWD/target/release"
export RUST_LOG=info
export MINIMAL_LOG_FORMAT=true

echo "✅ 写入 $PROFILE 成功，请执行：source $PROFILE"

# Step 9: 启动说明
echo -e "\n🚀 [9/9] 启动指引："
echo "➡️ Leader 节点：screen -S leader bash -c 'make run-nockchain-leader'"
echo "➡️ Follower 节点：screen -S follower bash -c 'make run-nockchain-follower'"
echo "💬 查看日志：screen -r leader 或 screen -r follower"
echo "🔙 退出 screen：按 Ctrl+A 然后按 D"
echo -e "\n📦 安装日志：$LOGFILE"
echo -e "\n🎉 安装完成，祝你挖矿愉快！"
