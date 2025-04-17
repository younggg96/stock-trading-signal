#!/usr/bin/env python3
"""
简单的 Tkinter 测试程序
"""
import tkinter as tk
from tkinter import ttk, messagebox
import os
import sys

# 设置 Tkinter 弃用警告静默
os.environ['TK_SILENCE_DEPRECATION'] = '1'

def main():
    print("创建 Tkinter 根窗口...")
    root = tk.Tk()
    print("设置窗口标题...")
    root.title("Tkinter 测试")
    print("设置窗口大小...")
    root.geometry("400x300")
    
    print("创建文本标签...")
    label = tk.Label(root, text="这是一个测试窗口", font=("Arial", 18))
    label.pack(pady=20)
    
    print("创建按钮...")
    button = ttk.Button(root, text="点击我", command=lambda: messagebox.showinfo("消息", "按钮被点击了!"))
    button.pack(pady=20)
    
    print("初始化完成，准备显示窗口...")
    
    # 确保窗口显示在最前面
    root.lift()
    root.attributes('-topmost', True)
    root.after_idle(root.attributes, '-topmost', False)

    # 检查平台
    if sys.platform == 'darwin':  # macOS
        print("检测到 macOS 系统，尝试额外设置...")
        try:
            # 尝试使用 macOS 特定命令使应用前置
            os.system('''/usr/bin/osascript -e 'tell app "Python" to activate' ''')
        except Exception as e:
            print(f"macOS 激活应用失败: {e}")
    
    print("启动 Tkinter 主循环...")
    root.mainloop()
    print("主循环结束")

if __name__ == "__main__":
    print("开始测试 Tkinter...")
    main()
    print("测试完成") 