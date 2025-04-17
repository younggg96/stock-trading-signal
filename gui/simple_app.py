#!/usr/bin/env python3
"""
简化版股票交易信号可视化 GUI 应用
只包含最基本的功能
"""
import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import numpy as np

# 设置 TK_SILENCE_DEPRECATION 环境变量
os.environ['TK_SILENCE_DEPRECATION'] = '1'

class SimpleStockApp:
    def __init__(self, root):
        self.root = root
        self.root.title("简易股票图表")
        self.root.geometry("800x600")
        
        # 创建主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建按钮
        self.button = ttk.Button(main_frame, text="生成示例图表", command=self.generate_chart)
        self.button.pack(pady=10)
        
        # 创建图表容器
        self.chart_frame = ttk.Frame(main_frame)
        self.chart_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    def generate_chart(self):
        # 生成示例数据
        dates = pd.date_range(start='2023-01-01', periods=30)
        close_prices = np.random.normal(100, 10, 30).cumsum() + 1000
        
        # 创建 DataFrame
        df = pd.DataFrame({
            'timestamp': dates,
            'close': close_prices
        })
        
        # 清空图表框架
        for widget in self.chart_frame.winfo_children():
            widget.destroy()
        
        # 创建图表
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 绘制收盘价
        ax.plot(df['timestamp'], df['close'], label='Close Price')
        
        # 设置标题和标签
        ax.set_title('Sample Stock Price Chart')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        
        # 旋转 x 轴标签
        plt.xticks(rotation=45)
        
        # 添加图例
        ax.legend()
        
        # 调整布局
        plt.tight_layout()
        
        # 将图表添加到 tkinter 窗口
        canvas = FigureCanvasTkAgg(fig, self.chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 显示成功消息
        messagebox.showinfo("成功", "示例图表已生成")

def main():
    root = tk.Tk()
    app = SimpleStockApp(root)
    root.mainloop()

if __name__ == "__main__":
    main() 