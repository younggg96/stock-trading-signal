"""
Vercel部署入口点文件
"""
import sys
import os

# 将根目录添加到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入应用服务器
from stock_web_app import server

# 暴露应用给Vercel
app = server 