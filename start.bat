@echo off
REM GTO v6.0 启动脚本 (Windows)

echo ==========================================
echo   GTO v6.0 足球预测系统
echo ==========================================

REM 检查Python
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误: 未找到Python
    pause
    exit /b 1
)

REM 启动API服务器
echo.
echo 启动API服务器...
echo 访问地址: http://localhost:8080
echo 按 Ctrl+C 停止
echo.

python -m src.api.server
pause
