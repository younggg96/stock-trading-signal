#!/usr/bin/env python3
"""
股票交易信号可视化 GUI 应用 - 调试版本
带有异常捕获和详细日志输出
"""
import os
import sys
import traceback
import logging

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("gui_debug.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("stock_gui_debug")

def main():
    try:
        # 尝试导入 tkinter
        logger.info("正在导入 tkinter...")
        import tkinter as tk
        logger.info("tkinter 导入成功")
        
        # 设置 TK_SILENCE_DEPRECATION 环境变量
        os.environ['TK_SILENCE_DEPRECATION'] = '1'
        
        # 尝试导入 matplotlib
        logger.info("正在导入 matplotlib...")
        import matplotlib
        matplotlib.use('TkAgg')  # 明确设置 matplotlib 后端
        import matplotlib.pyplot as plt
        logger.info(f"matplotlib 版本: {matplotlib.__version__}")
        
        # 尝试导入项目其他模块
        logger.info("正在导入项目模块...")
        # 添加项目根目录到系统路径
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.append(project_root)
        logger.info(f"项目根目录: {project_root}")
        
        # 尝试创建一个最小的 tkinter 窗口
        logger.info("正在创建测试 tkinter 窗口...")
        test_root = tk.Tk()
        test_root.title("测试窗口")
        test_root.geometry("200x100")
        tk.Label(test_root, text="如果你能看到这个窗口，tkinter 工作正常").pack(padx=20, pady=20)
        
        # 定义一个回调函数，在窗口打开后继续加载主应用
        def load_main_app():
            try:
                logger.info("测试窗口打开成功，正在关闭测试窗口...")
                test_root.destroy()
                
                logger.info("正在导入主应用模块...")
                from gui.stock_app import main
                
                logger.info("正在启动主应用...")
                main()
                
            except Exception as e:
                logger.error(f"加载主应用时出错: {str(e)}")
                logger.error(traceback.format_exc())
                
                # 显示错误窗口
                error_root = tk.Tk()
                error_root.title("应用加载错误")
                error_root.geometry("600x400")
                
                error_frame = tk.Frame(error_root, padx=20, pady=20)
                error_frame.pack(fill=tk.BOTH, expand=True)
                
                tk.Label(error_frame, text="应用加载过程中出现错误:", font=("Arial", 12, "bold")).pack(anchor=tk.W)
                tk.Label(error_frame, text=str(e), wraplength=550, justify=tk.LEFT).pack(fill=tk.X, pady=10)
                
                # 显示详细错误信息
                tk.Label(error_frame, text="详细错误信息:", font=("Arial", 12, "bold")).pack(anchor=tk.W, pady=(10, 5))
                
                error_text = tk.Text(error_frame, height=15)
                error_text.insert(tk.END, traceback.format_exc())
                error_text.config(state=tk.DISABLED)
                error_text.pack(fill=tk.BOTH, expand=True)
                
                tk.Button(error_frame, text="关闭", command=error_root.destroy).pack(pady=10)
                
                error_root.mainloop()
        
        # 1000ms 后执行回调函数（给测试窗口足够时间显示）
        test_root.after(1000, load_main_app)
        
        logger.info("开始 tkinter 主循环...")
        test_root.mainloop()
        
    except Exception as e:
        logger.error(f"初始化时出错: {str(e)}")
        logger.error(traceback.format_exc())
        
        # 如果 tkinter 可用，显示错误窗口
        try:
            import tkinter as tk
            from tkinter import scrolledtext
            
            error_root = tk.Tk()
            error_root.title("严重错误")
            error_root.geometry("600x400")
            
            error_frame = tk.Frame(error_root, padx=20, pady=20)
            error_frame.pack(fill=tk.BOTH, expand=True)
            
            tk.Label(error_frame, text="应用启动时出现严重错误:", font=("Arial", 12, "bold")).pack(anchor=tk.W)
            tk.Label(error_frame, text=str(e), wraplength=550, justify=tk.LEFT).pack(fill=tk.X, pady=10)
            
            # 显示详细错误信息
            tk.Label(error_frame, text="详细错误信息:", font=("Arial", 12, "bold")).pack(anchor=tk.W, pady=(10, 5))
            
            error_text = scrolledtext.ScrolledText(error_frame, height=15)
            error_text.insert(tk.END, traceback.format_exc())
            error_text.config(state=tk.DISABLED)
            error_text.pack(fill=tk.BOTH, expand=True)
            
            tk.Button(error_frame, text="关闭", command=error_root.destroy).pack(pady=10)
            
            error_root.mainloop()
            
        except Exception:
            # 如果 tkinter 也不可用，只能打印到控制台
            print("严重错误: " + str(e))
            print("\n详细错误信息:")
            traceback.print_exc()

if __name__ == "__main__":
    main() 