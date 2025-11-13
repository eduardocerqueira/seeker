#date: 2025-11-13T17:02:37Z
#url: https://api.github.com/gists/663998931618b8fa09d7899c7d7ed40f
#owner: https://api.github.com/users/wangwei334455

#!/bin/bash
# 统一后端启动脚本
# 一键启动所有后端组件

cd "$(dirname "$0")"

echo "=========================================="
echo "AI交易系统 - 统一后端启动"
echo "=========================================="
echo ""
echo "将启动以下服务："
echo "  1. API Server (Flask + Socket.IO)"
echo "  2. Data Puller (数据采集器)"
echo "  3. L2 Strategy Core (策略核心)"
echo "  4. OrderExecutor (订单执行器)"
echo ""
echo "按 Ctrl+C 停止所有服务"
echo "=========================================="
echo ""

# 🔴 修复：确保使用虚拟环境的Python
if [ -d "venv" ]; then
    # 激活虚拟环境并运行
    source venv/bin/activate
    python3 scripts/start_backend.py
else
    echo "⚠️  警告: 未找到虚拟环境，使用系统Python"
    python3 scripts/start_backend.py
fi

