#!/bin/bash
# GTO v6.0 启动脚本

echo "=========================================="
echo "  GTO v6.0 足球预测系统"
echo "=========================================="

# 检查Python
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到python3"
    exit 1
fi

# 进入项目目录
cd "$(dirname "$0")"

# 启动API服务器
echo ""
echo "启动API服务器..."
echo "访问地址: http://localhost:8080"
echo "按 Ctrl+C 停止"
echo ""

python3 -m src.api.server
