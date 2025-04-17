import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib
# 设置 matplotlib 后端，必须在导入 pyplot 之前
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import threading
import sys
import os
import datetime

# 设置 Tkinter 弃用警告静默
os.environ['TK_SILENCE_DEPRECATION'] = '1'

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_fetch.polygon_api import fetch_polygon_15min_data
from features.technical_indicators import compute_technical_indicators
from model.train_model import train_model
from model.signal_generator import generate_signals


class StockTradingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("股票交易信号可视化")
        self.root.geometry("1200x800")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # 确保窗口在屏幕中心
        self.center_window()
        
        self.setup_ui()
        self.data = None
        self.is_processing = False
    
    def center_window(self):
        """将窗口居中显示"""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry('{}x{}+{}+{}'.format(width, height, x, y))
        
    def setup_ui(self):
        # 创建主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建控制面板
        control_frame = ttk.LabelFrame(main_frame, text="控制面板", padding="10")
        control_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # 第一行：API 密钥和股票代码
        api_frame = ttk.Frame(control_frame)
        api_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(api_frame, text="API 密钥:").pack(side=tk.LEFT, padx=5)
        self.api_key_var = tk.StringVar(value=os.environ.get("POLYGON_API_KEY", ""))
        self.api_key_entry = ttk.Entry(api_frame, textvariable=self.api_key_var, width=40, show="*")
        self.api_key_entry.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(api_frame, text="股票代码:").pack(side=tk.LEFT, padx=5)
        self.symbol_var = tk.StringVar(value="AAPL")
        self.symbol_entry = ttk.Entry(api_frame, textvariable=self.symbol_var, width=10)
        self.symbol_entry.pack(side=tk.LEFT, padx=5)
        
        # 第二行：日期范围
        date_frame = ttk.Frame(control_frame)
        date_frame.pack(fill=tk.X, pady=5)
        
        # 默认日期范围为过去2天
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=2)
        
        ttk.Label(date_frame, text="开始日期:").pack(side=tk.LEFT, padx=5)
        self.start_date_var = tk.StringVar(value=start_date.strftime('%Y-%m-%d'))
        self.start_date_entry = ttk.Entry(date_frame, textvariable=self.start_date_var, width=12)
        self.start_date_entry.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(date_frame, text="结束日期:").pack(side=tk.LEFT, padx=5)
        self.end_date_var = tk.StringVar(value=end_date.strftime('%Y-%m-%d'))
        self.end_date_entry = ttk.Entry(date_frame, textvariable=self.end_date_var, width=12)
        self.end_date_entry.pack(side=tk.LEFT, padx=5)
        
        # 按钮区域
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        self.fetch_btn = ttk.Button(button_frame, text="获取数据并分析", command=self.fetch_and_analyze)
        self.fetch_btn.pack(side=tk.LEFT, padx=5)
        
        self.save_api_btn = ttk.Button(button_frame, text="保存 API 密钥", command=self.save_api_key)
        self.save_api_btn.pack(side=tk.LEFT, padx=5)

        # 添加测试按钮
        self.test_btn = ttk.Button(button_frame, text="测试图表显示", command=self.test_chart)
        self.test_btn.pack(side=tk.LEFT, padx=5)
        
        # 状态栏
        self.status_var = tk.StringVar(value="就绪")
        self.status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # 进度条
        self.progress = ttk.Progressbar(main_frame, orient=tk.HORIZONTAL, length=100, mode='indeterminate')
        self.progress.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)
        
        # 创建选项卡
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 股票价格图表选项卡
        self.price_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.price_frame, text="价格图表")
        
        # 技术指标选项卡
        self.indicators_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.indicators_frame, text="技术指标")
        
        # 交易信号选项卡
        self.signals_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.signals_frame, text="交易信号")
        
        # 数据表格选项卡
        self.data_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.data_frame, text="数据表格")
        
        # 创建表格
        self.setup_table(self.data_frame)
    
    def test_chart(self):
        """创建测试图表以验证显示功能"""
        try:
            # 生成示例数据
            dates = pd.date_range(start='2023-01-01', periods=30)
            close_prices = [100 + i * 2 + (i % 5) * 3 for i in range(30)]
            
            # 创建 DataFrame
            self.data = pd.DataFrame({
                'timestamp': dates,
                'close': close_prices,
                'ma_10': [sum(close_prices[max(0, i-9):i+1])/min(10, i+1) for i in range(30)],
                'signal': ['BUY' if i % 7 == 0 else 'HOLD' for i in range(30)]
            })
            
            # 计算 RSI 和 MACD (简化版)
            self.data['rsi'] = 50 + [15 * (i % 3 - 1) for i in range(30)]
            self.data['macd'] = [2 * ((i % 10) - 5) for i in range(30)]
            self.data['macd_signal'] = [1.5 * ((i % 12) - 6) for i in range(30)]
            
            # 更新 UI
            self.update_ui()
            
            # 显示成功消息
            messagebox.showinfo("成功", "测试图表已生成")
            
        except Exception as e:
            messagebox.showerror("错误", f"生成测试图表时出错: {str(e)}")
            import traceback
            traceback.print_exc()
        
    def setup_table(self, parent):
        # 创建表格容器和滚动条
        table_frame = ttk.Frame(parent)
        table_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        xscroll = ttk.Scrollbar(table_frame, orient=tk.HORIZONTAL)
        xscroll.pack(side=tk.BOTTOM, fill=tk.X)
        
        yscroll = ttk.Scrollbar(table_frame, orient=tk.VERTICAL)
        yscroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 创建表格
        columns = ('timestamp', 'open', 'high', 'low', 'close', 'volume', 'ma_10', 'rsi', 'macd', 'macd_signal', 'signal')
        self.treeview = ttk.Treeview(table_frame, columns=columns, show='headings',
                                    xscrollcommand=xscroll.set, yscrollcommand=yscroll.set)
        
        # 设置列标题
        for col in columns:
            self.treeview.heading(col, text=col.capitalize())
            # 根据列内容设置宽度
            if col == 'timestamp':
                self.treeview.column(col, width=150, anchor='center')
            elif col == 'signal':
                self.treeview.column(col, width=80, anchor='center')
            else:
                self.treeview.column(col, width=100, anchor='e')
        
        self.treeview.pack(fill=tk.BOTH, expand=True)
        
        # 配置滚动条
        xscroll.config(command=self.treeview.xview)
        yscroll.config(command=self.treeview.yview)
    
    def save_api_key(self):
        api_key = self.api_key_var.get().strip()
        if not api_key:
            messagebox.showerror("错误", "API 密钥不能为空")
            return
        
        # 将 API 密钥保存到环境变量
        os.environ["POLYGON_API_KEY"] = api_key
        
        # 提示用户如何永久保存
        messagebox.showinfo("成功", "API 密钥已保存到当前会话。\n\n要永久保存，请在终端运行：\n\nexport POLYGON_API_KEY='your_api_key' >> ~/.zshrc")
    
    def fetch_and_analyze(self):
        if self.is_processing:
            return
        
        # 获取输入值
        api_key = self.api_key_var.get().strip()
        symbol = self.symbol_var.get().strip().upper()
        start_date = self.start_date_var.get().strip()
        end_date = self.end_date_var.get().strip()
        
        # 验证输入
        if not api_key:
            messagebox.showerror("错误", "请输入 API 密钥")
            return
            
        if not symbol:
            messagebox.showerror("错误", "请输入股票代码")
            return
        
        # 开始处理
        self.is_processing = True
        self.status_var.set("正在获取数据...")
        self.progress.start()
        self.fetch_btn.state(['disabled'])
        
        # 清空表格和图表
        self.treeview.delete(*self.treeview.get_children())
        for widget in self.price_frame.winfo_children():
            widget.destroy()
        for widget in self.indicators_frame.winfo_children():
            widget.destroy()
        for widget in self.signals_frame.winfo_children():
            widget.destroy()
        
        # 在新线程中处理数据，避免 UI 卡顿
        thread = threading.Thread(target=self.process_data, args=(api_key, symbol, start_date, end_date))
        thread.daemon = True
        thread.start()
    
    def process_data(self, api_key, symbol, start_date, end_date):
        try:
            # 临时修改 polygon_api.py 的日期参数
            import data_fetch.polygon_api as polygon_api
            original_func = polygon_api.fetch_polygon_15min_data
            
            def custom_fetch(symbol, api_key, limit=1000):
                # 使用用户指定的日期
                from datetime import datetime
                
                # 解析日期
                try:
                    start_date_obj = datetime.strptime(start_date, '%Y-%m-%d')
                    end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
                except ValueError:
                    self.root.after(0, lambda: messagebox.showerror("错误", "日期格式无效，请使用 YYYY-MM-DD 格式"))
                    return pd.DataFrame()
                
                print(f"Requesting data from {start_date} to {end_date}")
                
                url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/15/minute/{start_date}/{end_date}"
                params = {
                    "adjusted": "true",
                    "sort": "asc",
                    "limit": limit,
                    "apiKey": api_key
                }
                
                # 继续使用原始函数的其余逻辑
                import inspect
                source_lines = inspect.getsource(original_func)
                
                # 找到 API 请求之后的代码
                request_line_idx = source_lines.find("response = requests.get(url, params=params)")
                if request_line_idx != -1:
                    # 执行请求部分
                    import requests
                    response = requests.get(url, params=params)
                    
                    # 继续处理响应
                    remaining_code = source_lines[request_line_idx + len("response = requests.get(url, params=params)"):]
                    # 提取处理响应的代码
                    response_processing = remaining_code
                    
                    # 执行处理响应的代码（这里只能使用最简单的方式）
                    print(f"API Status Code: {response.status_code}")
                    
                    data = response.json()
                    
                    print("API Response Keys:", list(data.keys()))
                    
                    if "results" not in data:
                        error_msg = data.get("error", data.get("message", "Unknown error"))
                        print("Error in API response:", error_msg)
                        print("Full response:", data)
                        raise ValueError(f"Failed to get data from Polygon API: {error_msg}")
                    
                    results = data.get("results", [])
                    
                    if not results:
                        print("No results returned from API")
                        return pd.DataFrame()
                    
                    if results:
                        print("First result structure:", list(results[0].keys()))
                    
                    df = pd.DataFrame(results)
                    
                    if 't' in df.columns:
                        df['t'] = pd.to_datetime(df['t'], unit='ms')
                        df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume", "t": "timestamp"})
                    else:
                        print("Column 't' not found in API response. Available columns:", df.columns.tolist())
                        timestamp_candidates = [col for col in df.columns if any(time_str in str(col).lower() for time_str in ['time', 'date', 'timestamp'])]
                        if timestamp_candidates:
                            print(f"Possible timestamp column found: {timestamp_candidates[0]}")
                            df['timestamp'] = pd.to_datetime(df[timestamp_candidates[0]], unit='ms')
                        
                        col_mapping = {"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
                        for old_col, new_col in col_mapping.items():
                            if old_col in df.columns:
                                df = df.rename(columns={old_col: new_col})
                    
                    return df
                else:
                    raise ValueError("Cannot parse the original function")
            
            # 替换原始函数
            polygon_api.fetch_polygon_15min_data = custom_fetch
            
            # 现在调用主流程
            try:
                # 获取数据
                df = fetch_polygon_15min_data(symbol, api_key)
                
                if df.empty:
                    self.root.after(0, lambda: messagebox.showerror("错误", "没有数据返回"))
                    return
                
                # 更新状态
                self.root.after(0, lambda: self.status_var.set("计算技术指标..."))
                
                # 计算指标
                df = compute_technical_indicators(df)
                
                # 更新状态
                self.root.after(0, lambda: self.status_var.set("生成交易信号..."))
                
                try:
                    # 训练模型（可能会失败，所以放在 try 中）
                    train_model(df)
                    
                    # 生成信号
                    df_with_signals = generate_signals(df)
                    
                    # 合并数据
                    df = df.merge(df_with_signals[['timestamp', 'signal']], on='timestamp', how='left')
                except Exception as e:
                    # 如果生成信号失败，添加一个空的信号列
                    print(f"Error generating signals: {str(e)}")
                    df['signal'] = 'N/A'
                
                # 存储数据
                self.data = df
                
                # 更新 UI
                self.root.after(0, self.update_ui)
                
            finally:
                # 恢复原始函数
                polygon_api.fetch_polygon_15min_data = original_func
        
        except Exception as e:
            # 更新状态并显示错误
            error_msg = str(e)
            print(f"Error: {error_msg}")
            self.root.after(0, lambda: self.status_var.set(f"错误: {error_msg[:50]}..."))
            self.root.after(0, lambda: messagebox.showerror("处理错误", f"处理数据时出错: {error_msg}"))
        
        finally:
            # 停止进度条并启用按钮
            self.root.after(0, self.progress.stop)
            self.root.after(0, lambda: self.fetch_btn.state(['!disabled']))
            self.is_processing = False
    
    def update_ui(self):
        try:
            # 更新表格
            self.update_table()
            
            # 更新价格图表
            self.create_price_chart()
            
            # 更新技术指标图表
            self.create_indicators_chart()
            
            # 更新交易信号图表
            self.create_signals_chart()
            
            # 更新状态
            self.status_var.set(f"就绪 - 已加载 {len(self.data)} 条数据")
        except Exception as e:
            messagebox.showerror("错误", f"更新 UI 时出错: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def update_table(self):
        # 清空表格
        self.treeview.delete(*self.treeview.get_children())
        
        # 添加数据
        for i, row in self.data.iterrows():
            values = []
            for col in self.treeview['columns']:
                if col in row:
                    if col == 'timestamp':
                        # 格式化时间戳
                        values.append(row[col].strftime('%Y-%m-%d %H:%M'))
                    elif col in ['ma_10', 'rsi', 'macd', 'macd_signal', 'open', 'high', 'low', 'close']:
                        # 格式化数值
                        try:
                            values.append(f"{row[col]:.2f}")
                        except:
                            values.append(str(row[col]))
                    else:
                        values.append(row[col])
                else:
                    values.append('')
            
            self.treeview.insert('', 'end', values=values)
    
    def create_price_chart(self):
        # 清空现有内容
        for widget in self.price_frame.winfo_children():
            widget.destroy()
            
        # 创建价格图表
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 绘制收盘价
        ax.plot(self.data['timestamp'], self.data['close'], label='Close Price')
        
        # 添加移动平均线
        if 'ma_10' in self.data.columns:
            ax.plot(self.data['timestamp'], self.data['ma_10'], label='10-period MA', linestyle='--')
        
        # 设置标题和标签
        ax.set_title(f'{self.symbol_var.get()} Price Chart')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        
        # 旋转 x 轴标签
        plt.xticks(rotation=45)
        
        # 添加图例
        ax.legend()
        
        # 调整布局
        plt.tight_layout()
        
        # 将图表添加到 tkinter 窗口
        canvas = FigureCanvasTkAgg(fig, self.price_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 添加工具栏
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        toolbar = NavigationToolbar2Tk(canvas, self.price_frame)
        toolbar.update()
    
    def create_indicators_chart(self):
        # 清空现有内容
        for widget in self.indicators_frame.winfo_children():
            widget.destroy()
            
        # 创建技术指标图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # 绘制 RSI
        if 'rsi' in self.data.columns:
            ax1.plot(self.data['timestamp'], self.data['rsi'], label='RSI', color='purple')
            ax1.axhline(y=70, color='r', linestyle='--')
            ax1.axhline(y=30, color='g', linestyle='--')
            ax1.set_title('RSI (14)')
            ax1.set_ylabel('RSI')
            ax1.set_ylim(0, 100)
            ax1.legend()
        
        # 绘制 MACD
        if 'macd' in self.data.columns and 'macd_signal' in self.data.columns:
            ax2.plot(self.data['timestamp'], self.data['macd'], label='MACD', color='blue')
            ax2.plot(self.data['timestamp'], self.data['macd_signal'], label='Signal Line', color='red')
            
            # 绘制 MACD 柱状图
            for i in range(len(self.data)):
                if i > 0:
                    color = 'g' if self.data['macd'].iloc[i] > self.data['macd_signal'].iloc[i] else 'r'
                    ax2.bar(self.data['timestamp'].iloc[i], 
                            self.data['macd'].iloc[i] - self.data['macd_signal'].iloc[i],
                            color=color, width=0.01)
            
            ax2.set_title('MACD')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('MACD')
            ax2.legend()
        
        # 旋转 x 轴标签
        plt.xticks(rotation=45)
        
        # 调整布局
        plt.tight_layout()
        
        # 将图表添加到 tkinter 窗口
        canvas = FigureCanvasTkAgg(fig, self.indicators_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 添加工具栏
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        toolbar = NavigationToolbar2Tk(canvas, self.indicators_frame)
        toolbar.update()
    
    def create_signals_chart(self):
        # 清空现有内容
        for widget in self.signals_frame.winfo_children():
            widget.destroy()
            
        # 创建交易信号图表
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 绘制收盘价
        ax.plot(self.data['timestamp'], self.data['close'], label='Close Price')
        
        # 添加交易信号
        if 'signal' in self.data.columns:
            # 买入信号
            buy_signals = self.data[self.data['signal'] == 'BUY']
            if not buy_signals.empty:
                ax.scatter(buy_signals['timestamp'], buy_signals['close'], 
                          color='green', label='Buy Signal', marker='^', s=100)
            
            # 卖出信号（如果有的话）
            sell_signals = self.data[self.data['signal'] == 'SELL']
            if not sell_signals.empty:
                ax.scatter(sell_signals['timestamp'], sell_signals['close'], 
                          color='red', label='Sell Signal', marker='v', s=100)
        
        # 设置标题和标签
        ax.set_title(f'{self.symbol_var.get()} Trading Signals')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        
        # 旋转 x 轴标签
        plt.xticks(rotation=45)
        
        # 添加图例
        ax.legend()
        
        # 调整布局
        plt.tight_layout()
        
        # 将图表添加到 tkinter 窗口
        canvas = FigureCanvasTkAgg(fig, self.signals_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 添加工具栏
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        toolbar = NavigationToolbar2Tk(canvas, self.signals_frame)
        toolbar.update()
    
    def on_closing(self):
        if messagebox.askokcancel("退出", "确定要退出应用程序吗?"):
            plt.close('all')  # 关闭所有 matplotlib 图表
            self.root.destroy()

def main():
    root = tk.Tk()
    app = StockTradingApp(root)
    root.mainloop()

if __name__ == "__main__":
    main() 