#!/usr/bin/env python3
"""
PyQt5 版本的股票交易信号可视化
作为 Tkinter 版本的替代方案
"""
import sys
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

print("请先安装 PyQt5:")
print("python3 -m pip install PyQt5")
print("如果已安装，请忽略此消息")

try:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
        QPushButton, QLabel, QLineEdit, QTabWidget, QFrame, QTableWidget, 
        QTableWidgetItem, QHeaderView, QMessageBox, QProgressBar, QStatusBar,
        QGroupBox, QFormLayout
    )
    from PyQt5.QtCore import Qt, QThread, pyqtSignal, QDate
    from PyQt5.QtGui import QFont
    
    class StockChartCanvas(FigureCanvas):
        """股票图表画布"""
        def __init__(self, parent=None, width=10, height=6, dpi=100):
            self.fig, self.ax = plt.subplots(figsize=(width, height), dpi=dpi)
            super().__init__(self.fig)
            self.setParent(parent)
            self.fig.tight_layout()
    
    class DataProcessingThread(QThread):
        """数据处理线程，避免 UI 卡顿"""
        finished = pyqtSignal(object)
        error = pyqtSignal(str)
        
        def __init__(self, parent=None):
            super().__init__(parent)
            self.data = None
            
        def run(self):
            try:
                # 生成示例数据
                dates = pd.date_range(start='2023-01-01', periods=30)
                close_prices = [100 + i * 2 + (i % 5) * 3 for i in range(30)]
                
                # 创建 DataFrame
                data = pd.DataFrame({
                    'timestamp': dates,
                    'close': close_prices,
                    'open': [close_prices[i] - 5 + (i % 5) for i in range(30)],
                    'high': [close_prices[i] + 10 - (i % 3) for i in range(30)],
                    'low': [close_prices[i] - 10 + (i % 4) for i in range(30)],
                    'volume': [100000 + 10000 * (i % 10) for i in range(30)],
                    'ma_10': [sum(close_prices[max(0, i-9):i+1])/min(10, i+1) for i in range(30)],
                    'signal': ['BUY' if i % 7 == 0 else 'HOLD' for i in range(30)]
                })
                
                # 计算 RSI 和 MACD (简化版) - 修复列表操作的错误
                rsi_values = []
                macd_values = []
                macd_signal_values = []
                
                for i in range(30):
                    # 生成 RSI 值 (范围: 0-100)
                    rsi_value = 50 + 15 * ((i % 3) - 1)
                    rsi_values.append(rsi_value)
                    
                    # 生成 MACD 值
                    macd_value = 2 * ((i % 10) - 5)
                    macd_values.append(macd_value)
                    
                    # 生成 MACD 信号线值
                    macd_signal_value = 1.5 * ((i % 12) - 6)
                    macd_signal_values.append(macd_signal_value)
                
                data['rsi'] = rsi_values
                data['macd'] = macd_values
                data['macd_signal'] = macd_signal_values
                
                # 模拟处理延迟
                self.sleep(1)
                
                self.finished.emit(data)
            except Exception as e:
                import traceback
                print(traceback.format_exc())
                self.error.emit(str(e))
    
    class StockTradingApp(QMainWindow):
        """股票交易信号可视化应用程序"""
        def __init__(self):
            super().__init__()
            self.initUI()
            self.data = None
            self.data_thread = None
            
        def initUI(self):
            """初始化 UI"""
            self.setWindowTitle("股票交易信号可视化 (PyQt 版本)")
            self.setGeometry(100, 100, 1200, 800)
            
            # 创建中央组件
            central_widget = QWidget(self)
            self.setCentralWidget(central_widget)
            main_layout = QVBoxLayout(central_widget)
            
            # 创建控制面板
            control_group = QGroupBox("控制面板")
            control_layout = QVBoxLayout(control_group)
            
            # API 密钥和股票代码
            form_layout = QFormLayout()
            
            self.api_key_input = QLineEdit()
            self.api_key_input.setEchoMode(QLineEdit.Password)
            api_key = os.environ.get("POLYGON_API_KEY", "")
            self.api_key_input.setText(api_key)
            form_layout.addRow("API 密钥:", self.api_key_input)
            
            self.symbol_input = QLineEdit("AAPL")
            form_layout.addRow("股票代码:", self.symbol_input)
            
            # 日期选择
            date_layout = QHBoxLayout()
            self.start_date = QLineEdit()
            self.end_date = QLineEdit()
            
            # 设置默认日期为过去两天
            today = QDate.currentDate()
            two_days_ago = today.addDays(-2)
            self.start_date.setText(two_days_ago.toString("yyyy-MM-dd"))
            self.end_date.setText(today.toString("yyyy-MM-dd"))
            
            form_layout.addRow("开始日期:", self.start_date)
            form_layout.addRow("结束日期:", self.end_date)
            
            control_layout.addLayout(form_layout)
            
            # 按钮区域
            button_layout = QHBoxLayout()
            
            self.fetch_btn = QPushButton("获取数据并分析")
            self.fetch_btn.clicked.connect(self.fetchData)
            button_layout.addWidget(self.fetch_btn)
            
            self.save_api_btn = QPushButton("保存 API 密钥")
            self.save_api_btn.clicked.connect(self.saveApiKey)
            button_layout.addWidget(self.save_api_btn)

            self.test_btn = QPushButton("生成测试数据")
            self.test_btn.clicked.connect(self.generateTestData)
            button_layout.addWidget(self.test_btn)
            
            button_layout.addStretch()
            control_layout.addLayout(button_layout)
            
            main_layout.addWidget(control_group)
            
            # 状态栏和进度条
            self.status_bar = QStatusBar()
            self.setStatusBar(self.status_bar)
            self.status_bar.showMessage("就绪")
            
            self.progress_bar = QProgressBar()
            self.progress_bar.setVisible(False)
            main_layout.addWidget(self.progress_bar)
            
            # 创建选项卡
            self.tab_widget = QTabWidget()
            main_layout.addWidget(self.tab_widget)
            
            # 价格图表选项卡
            self.price_tab = QWidget()
            price_layout = QVBoxLayout(self.price_tab)
            self.price_canvas = StockChartCanvas(self.price_tab)
            price_layout.addWidget(self.price_canvas)
            self.price_toolbar = NavigationToolbar(self.price_canvas, self.price_tab)
            price_layout.addWidget(self.price_toolbar)
            self.tab_widget.addTab(self.price_tab, "价格图表")
            
            # 技术指标选项卡
            self.indicators_tab = QWidget()
            indicators_layout = QVBoxLayout(self.indicators_tab)
            self.indicators_canvas = StockChartCanvas(self.indicators_tab, height=8)
            indicators_layout.addWidget(self.indicators_canvas)
            self.indicators_toolbar = NavigationToolbar(self.indicators_canvas, self.indicators_tab)
            indicators_layout.addWidget(self.indicators_toolbar)
            self.tab_widget.addTab(self.indicators_tab, "技术指标")
            
            # 交易信号选项卡
            self.signals_tab = QWidget()
            signals_layout = QVBoxLayout(self.signals_tab)
            self.signals_canvas = StockChartCanvas(self.signals_tab)
            signals_layout.addWidget(self.signals_canvas)
            self.signals_toolbar = NavigationToolbar(self.signals_canvas, self.signals_tab)
            signals_layout.addWidget(self.signals_toolbar)
            self.tab_widget.addTab(self.signals_tab, "交易信号")
            
            # 数据表格选项卡
            self.data_tab = QWidget()
            data_layout = QVBoxLayout(self.data_tab)
            self.table = QTableWidget()
            data_layout.addWidget(self.table)
            self.tab_widget.addTab(self.data_tab, "数据表格")
            
            # 设置表格列
            self.setupTable()
        
        def setupTable(self):
            """设置数据表格"""
            columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                      'ma_10', 'rsi', 'macd', 'macd_signal', 'signal']
            self.table.setColumnCount(len(columns))
            self.table.setHorizontalHeaderLabels([col.capitalize() for col in columns])
            
            # 设置列宽
            header = self.table.horizontalHeader()
            header.setSectionResizeMode(0, QHeaderView.Stretch)  # timestamp 列
            for i in range(1, len(columns)):
                header.setSectionResizeMode(i, QHeaderView.ResizeToContents)
        
        def saveApiKey(self):
            """保存 API 密钥到环境变量"""
            api_key = self.api_key_input.text().strip()
            if not api_key:
                QMessageBox.warning(self, "错误", "API 密钥不能为空")
                return
            
            # 将 API 密钥保存到环境变量
            os.environ["POLYGON_API_KEY"] = api_key
            
            # 提示用户如何永久保存
            QMessageBox.information(self, "成功", 
                                "API 密钥已保存到当前会话。\n\n"
                                "要永久保存，请在终端运行：\n\n"
                                "export POLYGON_API_KEY='your_api_key' >> ~/.zshrc")
        
        def fetchData(self):
            """获取数据并分析"""
            # TODO: 实现真实的 API 数据获取
            QMessageBox.information(self, "提示", "此功能尚未实现，请先使用'生成测试数据'按钮")
        
        def generateTestData(self):
            """生成测试数据"""
            if self.data_thread is not None and self.data_thread.isRunning():
                return
            
            # 启用进度条
            self.progress_bar.setRange(0, 0)  # 不确定模式
            self.progress_bar.setVisible(True)
            self.status_bar.showMessage("正在生成测试数据...")
            self.fetch_btn.setEnabled(False)
            self.test_btn.setEnabled(False)
            
            # 在后台线程中生成数据
            self.data_thread = DataProcessingThread(self)
            self.data_thread.finished.connect(self.onDataProcessed)
            self.data_thread.error.connect(self.onDataError)
            self.data_thread.start()
        
        def onDataProcessed(self, data):
            """数据处理完成的回调"""
            self.data = data
            self.updateUI()
            
            # 恢复 UI 状态
            self.progress_bar.setVisible(False)
            self.status_bar.showMessage(f"就绪 - 已加载 {len(data)} 条数据")
            self.fetch_btn.setEnabled(True)
            self.test_btn.setEnabled(True)
        
        def onDataError(self, error_msg):
            """数据处理错误的回调"""
            QMessageBox.critical(self, "错误", f"生成数据时出错: {error_msg}")
            
            # 恢复 UI 状态
            self.progress_bar.setVisible(False)
            self.status_bar.showMessage("错误")
            self.fetch_btn.setEnabled(True)
            self.test_btn.setEnabled(True)
        
        def updateUI(self):
            """更新 UI 显示"""
            if self.data is None:
                return
            
            self.updateTable()
            self.createPriceChart()
            self.createIndicatorsChart()
            self.createSignalsChart()
        
        def updateTable(self):
            """更新数据表格"""
            # 清空表格
            self.table.setRowCount(0)
            
            # 添加数据
            self.table.setRowCount(len(self.data))
            
            # 获取列标题
            columns = []
            for j in range(self.table.columnCount()):
                columns.append(self.table.horizontalHeaderItem(j).text().lower())
            
            for i, (_, row) in enumerate(self.data.iterrows()):
                for j, col in enumerate(columns):
                    if col in row:
                        value = row[col]
                        
                        # 格式化值
                        if col == 'timestamp':
                            value = value.strftime('%Y-%m-%d %H:%M')
                        elif col in ['open', 'high', 'low', 'close', 'ma_10', 'rsi', 'macd', 'macd_signal']:
                            value = f"{value:.2f}"
                        
                        item = QTableWidgetItem(str(value))
                        
                        # 设置对齐方式
                        if col in ['signal']:
                            item.setTextAlignment(Qt.AlignCenter)
                        elif col == 'timestamp':
                            item.setTextAlignment(Qt.AlignCenter)
                        else:
                            item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                        
                        self.table.setItem(i, j, item)
        
        def createPriceChart(self):
            """创建价格图表"""
            # 重置图表
            self.price_canvas.ax.clear()
            
            # 绘制收盘价
            self.price_canvas.ax.plot(self.data['timestamp'], self.data['close'], label='收盘价')
            
            # 添加移动平均线
            if 'ma_10' in self.data.columns:
                self.price_canvas.ax.plot(self.data['timestamp'], self.data['ma_10'], 
                                        label='10周期移动平均线', linestyle='--')
            
            # 设置标题和标签
            symbol = self.symbol_input.text().strip().upper()
            self.price_canvas.ax.set_title(f'{symbol} 价格图表')
            self.price_canvas.ax.set_xlabel('日期')
            self.price_canvas.ax.set_ylabel('价格')
            
            # 旋转 x 轴标签
            plt.setp(self.price_canvas.ax.get_xticklabels(), rotation=45)
            
            # 添加图例
            self.price_canvas.ax.legend()
            
            # 调整布局
            self.price_canvas.fig.tight_layout()
            self.price_canvas.draw()
        
        def createIndicatorsChart(self):
            """创建技术指标图表"""
            # 重置图表
            self.indicators_canvas.fig.clear()
            
            # 创建两个子图
            ax1 = self.indicators_canvas.fig.add_subplot(211)  # RSI
            ax2 = self.indicators_canvas.fig.add_subplot(212, sharex=ax1)  # MACD
            
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
                ax2.plot(self.data['timestamp'], self.data['macd_signal'], 
                        label='信号线', color='red')
                
                # 绘制 MACD 柱状图
                for i in range(len(self.data)):
                    if i > 0:
                        color = 'g' if self.data['macd'].iloc[i] > self.data['macd_signal'].iloc[i] else 'r'
                        ax2.bar(self.data['timestamp'].iloc[i], 
                               self.data['macd'].iloc[i] - self.data['macd_signal'].iloc[i],
                               color=color, width=0.8)
                
                ax2.set_title('MACD')
                ax2.set_xlabel('日期')
                ax2.set_ylabel('MACD')
                ax2.legend()
            
            # 旋转 x 轴标签
            plt.setp(ax2.get_xticklabels(), rotation=45)
            
            # 调整布局
            self.indicators_canvas.fig.tight_layout()
            self.indicators_canvas.draw()
        
        def createSignalsChart(self):
            """创建交易信号图表"""
            # 重置图表
            self.signals_canvas.ax.clear()
            
            # 绘制收盘价
            self.signals_canvas.ax.plot(self.data['timestamp'], self.data['close'], label='收盘价')
            
            # 添加交易信号
            if 'signal' in self.data.columns:
                # 买入信号
                buy_signals = self.data[self.data['signal'] == 'BUY']
                if not buy_signals.empty:
                    self.signals_canvas.ax.scatter(buy_signals['timestamp'], buy_signals['close'], 
                                                color='green', label='买入信号', marker='^', s=100)
                
                # 卖出信号（如果有的话）
                sell_signals = self.data[self.data['signal'] == 'SELL']
                if not sell_signals.empty:
                    self.signals_canvas.ax.scatter(sell_signals['timestamp'], sell_signals['close'], 
                                                color='red', label='卖出信号', marker='v', s=100)
            
            # 设置标题和标签
            symbol = self.symbol_input.text().strip().upper()
            self.signals_canvas.ax.set_title(f'{symbol} 交易信号')
            self.signals_canvas.ax.set_xlabel('日期')
            self.signals_canvas.ax.set_ylabel('价格')
            
            # 旋转 x 轴标签
            plt.setp(self.signals_canvas.ax.get_xticklabels(), rotation=45)
            
            # 添加图例
            self.signals_canvas.ax.legend()
            
            # 调整布局
            self.signals_canvas.fig.tight_layout()
            self.signals_canvas.draw()

    def main():
        app = QApplication(sys.argv)
        main_window = StockTradingApp()
        main_window.show()
        sys.exit(app.exec_())

    if __name__ == "__main__":
        main()
        
except ImportError as e:
    print(f"\n错误: {e}")
    print("\n请先安装 PyQt5:")
    print("python3 -m pip install PyQt5") 