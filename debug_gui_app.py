#!/usr/bin/env python3
"""
启动调试版股票交易信号可视化 GUI 应用
"""
import os
import sys

# 设置 TK_SILENCE_DEPRECATION 环境变量
os.environ['TK_SILENCE_DEPRECATION'] = '1'

# 添加项目根目录到系统路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

# 导入调试版应用
from gui.debug_app import main

if __name__ == "__main__":
    print("正在启动调试版 GUI 应用...")
    try:
        main()
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc() 