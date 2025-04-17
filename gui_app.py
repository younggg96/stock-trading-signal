#!/usr/bin/env python3
"""
股票交易信号可视化 GUI 应用
"""
import os
import sys
import traceback

# 设置 Tkinter 弃用警告静默
os.environ['TK_SILENCE_DEPRECATION'] = '1'

# 添加项目根目录到系统路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

print(f"正在启动股票交易信号可视化 GUI 应用...")
print(f"Python 版本: {sys.version}")
print(f"工作目录: {os.getcwd()}")

try:
    import matplotlib
    # 设置 matplotlib 后端，必须在导入 pyplot 之前
    matplotlib.use('TkAgg')
    print(f"Matplotlib 版本: {matplotlib.__version__}")
    
    import tkinter as tk
    tk_version = tk.Tcl().eval('info patchlevel')
    print(f"Tkinter 版本: {tk_version}")
    
    from gui.stock_app import main
    
    if __name__ == "__main__":
        try:
            main()
        except Exception as e:
            print(f"运行主应用时出错: {e}")
            traceback.print_exc()
            
            # 如果主应用崩溃，显示错误窗口
            try:
                error_root = tk.Tk()
                error_root.title("应用错误")
                error_root.geometry("600x400")
                
                tk.Label(error_root, text="应用运行过程中发生错误:", font=("Arial", 14, "bold")).pack(pady=10)
                tk.Label(error_root, text=str(e), wraplength=550).pack(pady=10)
                
                # 显示详细错误信息
                tk.Label(error_root, text="详细错误信息:").pack(anchor=tk.W, padx=20)
                
                from tkinter import scrolledtext
                error_text = scrolledtext.ScrolledText(error_root, height=15, width=70)
                error_text.insert(tk.END, traceback.format_exc())
                error_text.config(state=tk.DISABLED)
                error_text.pack(padx=20, pady=10)
                
                tk.Button(error_root, text="关闭", command=error_root.destroy).pack(pady=10)
                
                error_root.mainloop()
            except Exception:
                # 如果错误窗口也失败，只能打印到控制台
                print("无法显示错误窗口")
            
except Exception as e:
    print(f"初始化应用时出错: {e}")
    traceback.print_exc()
    
    # 尝试显示简单的错误消息
    try:
        import tkinter as tk
        from tkinter import messagebox
        
        root = tk.Tk()
        root.withdraw()  # 隐藏主窗口
        
        messagebox.showerror("严重错误", f"应用初始化失败: {str(e)}\n\n请查看控制台获取详细信息。")
        
        root.destroy()
    except:
        print("无法显示错误对话框，请查看上面的错误信息。") 