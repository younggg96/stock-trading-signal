#!/usr/bin/env python3
"""
简化版股票交易信号可视化 - 使用 PyQt5
适用于 macOS 或其他 Tkinter 显示有问题的系统
"""
import sys
import os
import pandas as pd
import numpy as np
import datetime
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

try:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
        QPushButton, QLabel, QTabWidget, QTableWidget, QTableWidgetItem,
        QHeaderView, QMessageBox, QStatusBar, QLineEdit, QProgressBar,
        QComboBox, QCheckBox, QRadioButton, QButtonGroup, QGroupBox, QFormLayout,
        QAbstractItemView
    )
    from PyQt5.QtCore import Qt, QThread, pyqtSignal, QDateTime, QTimer
    from PyQt5.QtGui import QColor
    
    import matplotlib
    matplotlib.use('Qt5Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    
    # 配置中文字体支持
    from matplotlib import rcParams
    import platform
    
    # 根据操作系统选择合适的中文字体
    system = platform.system()
    if system == 'Windows':
        font_name = 'SimHei'  # Windows的黑体
    elif system == 'Darwin':  # macOS
        font_name = 'Arial Unicode MS'  # macOS常见中文字体
    else:  # Linux或其他
        font_name = 'WenQuanYi Zen Hei'  # Linux常用中文字体
    
    # 设置全局字体
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = [font_name, 'DejaVu Sans', 'Microsoft YaHei', 'SimSun']
    rcParams['axes.unicode_minus'] = False  # 正确显示负号
    
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
    
    class StockDataThread(QThread):
        """股票数据获取线程"""
        finished = pyqtSignal(object)
        error = pyqtSignal(str)
        progress = pyqtSignal(int)
        
        def __init__(self, stock_code, period='1y'):
            super().__init__()
            self.stock_code = stock_code
            self.period = period
            
        def run(self):
            try:
                # 更新进度
                self.progress.emit(10)
                
                # 从Yahoo Finance获取数据
                stock = yf.Ticker(self.stock_code)
                data = stock.history(period=self.period)
                
                if data.empty:
                    self.error.emit(f"无法获取股票 {self.stock_code} 的数据")
                    return
                
                # 更新进度
                self.progress.emit(50)
                
                # 确保索引是日期格式
                data.index = pd.to_datetime(data.index)
                
                # 计算技术指标
                # 移动平均线
                data['MA5'] = data['Close'].rolling(window=5).mean()
                data['MA20'] = data['Close'].rolling(window=20).mean()
                
                # RSI
                delta = data['Close'].diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                avg_gain = gain.rolling(window=14).mean()
                avg_loss = loss.rolling(window=14).mean()
                rs = avg_gain / avg_loss
                data['RSI'] = 100 - (100 / (1 + rs))
                
                # MACD
                data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()
                data['EMA26'] = data['Close'].ewm(span=26, adjust=False).mean()
                data['MACD'] = data['EMA12'] - data['EMA26']
                data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
                
                # 生成交易信号
                data['Signal'] = 'HOLD'  # 默认为持有
                
                # 基于均线交叉产生信号
                for i in range(1, len(data)):
                    if data['MA5'].iloc[i] > data['MA20'].iloc[i] and data['MA5'].iloc[i-1] <= data['MA20'].iloc[i-1]:
                        data.loc[data.index[i], 'Signal'] = 'BUY'
                    elif data['MA5'].iloc[i] < data['MA20'].iloc[i] and data['MA5'].iloc[i-1] >= data['MA20'].iloc[i-1]:
                        data.loc[data.index[i], 'Signal'] = 'SELL'
                
                # 更新进度
                self.progress.emit(90)
                
                # 返回数据
                self.finished.emit(data)
                
            except Exception as e:
                self.error.emit(str(e))
    
    class PredictionThread(QThread):
        """LSTM预测线程"""
        finished = pyqtSignal(object)
        error = pyqtSignal(str)
        progress = pyqtSignal(int)
        
        def __init__(self, data, days_to_predict=7, feature_columns=['Close'], stock_code="AAPL"):
            super().__init__()
            self.data = data
            self.days_to_predict = days_to_predict
            self.feature_columns = feature_columns
            self.stock_code = stock_code
            
        def create_features(self, df):
            """创建更多技术指标特征"""
            data = df.copy()
            
            # 确保索引是日期
            if not isinstance(data.index, pd.DatetimeIndex):
                data.index = pd.to_datetime(data.index)
            
            # 价格相关特征
            data['Return'] = data['Close'].pct_change()  # 日收益率
            data['Return_Lag1'] = data['Return'].shift(1)  # 前一天收益率
            
            # 添加移动平均线特征
            data['MA5'] = data['Close'].rolling(window=5).mean()
            data['MA10'] = data['Close'].rolling(window=10).mean()
            data['MA20'] = data['Close'].rolling(window=20).mean()
            
            # 添加MACD指标
            data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()
            data['EMA26'] = data['Close'].ewm(span=26, adjust=False).mean()
            data['MACD'] = data['EMA12'] - data['EMA26']
            data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
            data['MACD_Hist'] = data['MACD'] - data['Signal_Line']
            
            # 添加RSI指标
            delta = data['Close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            data['RSI'] = 100 - (100 / (1 + rs))
            
            # 添加布林带
            data['Middle_Band'] = data['Close'].rolling(window=20).mean()
            rolling_std = data['Close'].rolling(window=20).std()
            data['Upper_Band'] = data['Middle_Band'] + (rolling_std * 2)
            data['Lower_Band'] = data['Middle_Band'] - (rolling_std * 2)
            data['BB_Width'] = (data['Upper_Band'] - data['Lower_Band']) / data['Middle_Band']
            
            # 波动率特征
            data['Volatility'] = data['Return'].rolling(window=10).std()
            
            # 添加交易量特征
            if 'Volume' in data.columns:
                data['Volume_Change'] = data['Volume'].pct_change()
                data['Volume_MA5'] = data['Volume'].rolling(window=5).mean()
                data['Volume_Ratio'] = data['Volume'] / data['Volume_MA5']
            
            # 价格与移动平均线的关系
            data['Price_to_MA5'] = data['Close'] / data['MA5'] - 1
            data['Price_to_MA10'] = data['Close'] / data['MA10'] - 1
            data['Price_to_MA20'] = data['Close'] / data['MA20'] - 1
            
            # 添加周期性特征
            data['Day_of_Week'] = data.index.dayofweek
            data['Month'] = data.index.month
            
            # 删除NaN值
            data = data.dropna()
            
            return data
            
        def run(self):
            try:
                # 进度更新
                self.progress.emit(10)
                
                # 提取特征
                raw_data = self.data.copy()
                
                # 检查数据量是否足够
                if len(raw_data) < 100:  # 至少需要100天的数据
                    self.error.emit("数据量不足，需要至少100天的历史数据进行预测")
                    return
                
                # 创建高级特征
                feature_data = self.create_features(raw_data)
                self.progress.emit(15)
                
                # 根据股票代码调整模型参数
                # 为不同类型的股票设置不同的学习参数
                units1 = 64  # 第一层LSTM单元数
                units2 = 32  # 第二层LSTM单元数
                dropout_rate = 0.2  # 默认丢弃率
                epochs = 50  # 增加训练轮数
                batch_size = 32  # 默认批次大小
                patience = 5  # 早停轮数
                
                # 分析股票波动性
                returns = raw_data['Close'].pct_change().dropna()
                volatility = returns.std()
                
                # 根据股票代码和波动性调整参数
                if self.stock_code in ["TSLA", "NVDA", "COIN"]:  # 高波动性科技股
                    units1 = 128  # 增加模型复杂度
                    units2 = 64
                    dropout_rate = 0.3  # 增加丢弃率防止过拟合
                    epochs = 80  # 增加训练轮数
                    patience = 8
                    print(f"为高波动性股票 {self.stock_code} 增强训练参数")
                elif self.stock_code in ["AAPL", "MSFT", "GOOGL"]:  # 大型科技股
                    units1 = 96
                    units2 = 48
                    epochs = 60
                    patience = 6
                    print(f"为大型科技股 {self.stock_code} 增强训练参数")
                elif volatility > 0.02:  # 高波动性股票
                    units1 = 96
                    units2 = 48
                    dropout_rate = 0.25
                    epochs = 60
                    patience = 6
                    print(f"为高波动性股票增强训练参数 (波动率: {volatility:.4f})")
                elif volatility < 0.01:  # 低波动性股票
                    units1 = 48
                    units2 = 24
                    epochs = 40
                    patience = 4
                    print(f"为低波动性股票增强训练参数 (波动率: {volatility:.4f})")
                    
                # 选择特征列
                # 基础价格特征始终包含
                used_features = ['Close', 'Return', 'MA5', 'MA20', 'RSI', 'MACD', 'BB_Width']
                
                # 如果有交易量数据，添加交易量特征
                if 'Volume' in feature_data.columns:
                    volume_features = ['Volume', 'Volume_Ratio']
                    used_features.extend(volume_features)
                
                # 准备训练数据
                selected_data = feature_data[used_features].copy()
                print(f"使用特征: {used_features}")
                
                # 数据规范化
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_data = scaler.fit_transform(selected_data)
                
                # 准备LSTM输入序列
                X = []
                y = []
                sequence_length = min(60, len(scaled_data) // 3)  # 使用过去60天或数据量的三分之一
                
                if len(scaled_data) <= sequence_length:
                    self.error.emit(f"数据量不足，无法进行预测，需要至少 {sequence_length*3} 天数据")
                    return
                
                for i in range(sequence_length, len(scaled_data)):
                    X.append(scaled_data[i-sequence_length:i])
                    y.append(scaled_data[i, 0])  # 预测收盘价
                
                X, y = np.array(X), np.array(y)
                
                # 进度更新
                self.progress.emit(20)
                
                # 划分训练集和验证集
                split = int(0.8 * len(X))
                X_train, X_val = X[:split], X[split:]
                y_train, y_val = y[:split], y[split:]
                
                # 创建更复杂的LSTM模型
                from tensorflow.keras.callbacks import EarlyStopping
                
                model = Sequential()
                model.add(LSTM(units=units1, return_sequences=True, 
                             input_shape=(X_train.shape[1], X_train.shape[2])))
                model.add(Dropout(dropout_rate))
                model.add(LSTM(units=units2, return_sequences=False))
                model.add(Dropout(dropout_rate))
                model.add(Dense(units=16, activation='relu'))
                model.add(Dense(units=1))
                
                model.compile(optimizer='adam', loss='mean_squared_error')
                
                # 进度更新
                self.progress.emit(30)
                
                # 设置早停回调以防止过拟合
                early_stopping = EarlyStopping(
                    monitor='val_loss',
                    patience=patience,
                    restore_best_weights=True,
                    verbose=0
                )
                
                # 训练模型
                history = model.fit(
                    X_train, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(X_val, y_val),
                    callbacks=[early_stopping],
                    verbose=0
                )
                
                # 进度更新
                self.progress.emit(60)
                
                # 输出模型训练信息
                final_loss = history.history['loss'][-1]
                val_loss = history.history['val_loss'][-1]
                epochs_used = len(history.history['loss'])
                print(f"模型训练完成: {epochs_used} 轮, 损失: {final_loss:.4f}, 验证损失: {val_loss:.4f}")
                
                # 准备预测输入
                try:
                    last_sequence = scaled_data[-sequence_length:]
                    next_inputs = last_sequence.reshape(1, sequence_length, len(used_features))
                    
                    # 预测未来价格 - 使用迭代法
                    predicted_scaled = []
                    current_input = next_inputs.copy()
                    
                    for _ in range(self.days_to_predict):
                        next_pred = model.predict(current_input, verbose=0)[0, 0]
                        predicted_scaled.append(next_pred)
                        
                        # 更新输入序列 - 更复杂的特征更新
                        # 创建新的特征行
                        new_row = np.zeros((1, 1, len(used_features)))
                        new_row[0, 0, 0] = next_pred  # 收盘价
                        
                        # 其他特征采用简化近似值（实际应用中会更精确计算这些特征）
                        if len(used_features) > 1:
                            # 如果有足够的预测值，计算最近收益率
                            if len(predicted_scaled) > 1:
                                new_row[0, 0, 1] = predicted_scaled[-1] / predicted_scaled[-2] - 1  # Return
                            else:
                                new_row[0, 0, 1] = 0  # 默认Return
                            
                            # 简化的移动平均线
                            if len(predicted_scaled) >= 5:
                                new_row[0, 0, 2] = np.mean(predicted_scaled[-5:])  # MA5
                            else:
                                new_row[0, 0, 2] = next_pred  # MA5 近似
                                
                            # 其他特征使用最后已知值
                            for j in range(3, len(used_features)):
                                new_row[0, 0, j] = current_input[0, -1, j]
                        
                        # 移除最早的时间步，添加新的预测
                        current_input = np.append(current_input[:, 1:, :], new_row, axis=1)
                    
                    # 反向转换预测结果
                    temp_array = np.zeros((len(predicted_scaled), len(used_features)))
                    temp_array[:, 0] = np.array(predicted_scaled)  # 只设置第一列（收盘价）
                    
                    # 反归一化
                    predicted_full = scaler.inverse_transform(temp_array)
                    predicted_close = predicted_full[:, 0].reshape(-1, 1)  # 只取收盘价
                    
                except Exception as inner_e:
                    import traceback
                    print(f"预测错误详情: {traceback.format_exc()}")
                    self.error.emit(f"预测过程中发生错误: {str(inner_e)}")
                    return
                
                # 创建预测日期，跳过周末
                last_date = self.data.index[-1]
                future_dates = []
                current_date = last_date
                
                for _ in range(self.days_to_predict):
                    current_date = current_date + datetime.timedelta(days=1)
                    # 跳过周末 (5=Saturday, 6=Sunday)
                    while current_date.weekday() > 4:  # 5 = Saturday, 6 = Sunday
                        current_date = current_date + datetime.timedelta(days=1)
                    future_dates.append(current_date)
                
                # 创建预测数据框
                future_df = pd.DataFrame(index=future_dates)
                future_df['Predicted_Close'] = predicted_close.flatten()
                
                # 检查预测值合理性
                last_close = float(self.data['Close'].iloc[-1])
                min_reasonable = last_close * 0.7  # 最低合理值，不低于最后一个收盘价的70%
                max_reasonable = last_close * 1.3  # 最高合理值，不高于最后一个收盘价的130%
                
                # 如果预测值超出合理范围，调整预测值
                for i in range(len(future_df)):
                    if future_df['Predicted_Close'].iloc[i] < min_reasonable:
                        future_df.loc[future_df.index[i], 'Predicted_Close'] = min_reasonable
                    elif future_df['Predicted_Close'].iloc[i] > max_reasonable:
                        future_df.loc[future_df.index[i], 'Predicted_Close'] = max_reasonable
                
                # 计算移动平均线
                # 合并历史和预测数据
                combined_close = pd.concat([
                    self.data['Close'][-20:], 
                    future_df['Predicted_Close']
                ])
                
                # 计算5日移动平均线
                future_df['MA5'] = combined_close.rolling(window=5).mean().iloc[-self.days_to_predict:]
                
                # 计算预测准确性评估
                if len(y_val) > 0:
                    val_pred = model.predict(X_val, verbose=0)
                    val_pred_unscaled = scaler.inverse_transform(
                        np.hstack([val_pred, np.zeros((len(val_pred), len(used_features)-1))])
                    )[:, 0]
                    y_val_unscaled = scaler.inverse_transform(
                        np.hstack([y_val.reshape(-1, 1), np.zeros((len(y_val), len(used_features)-1))])
                    )[:, 0]
                    
                    # 计算RMSE (均方根误差)
                    rmse = np.sqrt(np.mean((val_pred_unscaled - y_val_unscaled)**2))
                    # 计算MAPE (平均绝对百分比误差)
                    mape = np.mean(np.abs((val_pred_unscaled - y_val_unscaled) / y_val_unscaled)) * 100
                    
                    print(f"验证集评估 - RMSE: {rmse:.4f}, MAPE: {mape:.2f}%")
                    
                    # 添加评估指标到预测数据框
                    future_df.attrs['rmse'] = rmse
                    future_df.attrs['mape'] = mape
                
                # 生成预测交易信号
                future_df['Signal'] = 'HOLD'
                
                # 对预测数据应用增强的规则
                for i in range(1, len(future_df)):
                    price_change = future_df['Predicted_Close'].iloc[i] - future_df['Predicted_Close'].iloc[i-1]
                    price_change_pct = price_change / future_df['Predicted_Close'].iloc[i-1]
                    
                    # 使用动态阈值，根据股票的历史波动性调整
                    buy_threshold = max(0.01, volatility * 0.5)
                    sell_threshold = min(-0.01, -volatility * 0.5)
                    
                    if price_change_pct > buy_threshold and future_df['Predicted_Close'].iloc[i] > future_df['MA5'].iloc[i]:
                        future_df.loc[future_df.index[i], 'Signal'] = 'BUY'
                    elif price_change_pct < sell_threshold and future_df['Predicted_Close'].iloc[i] < future_df['MA5'].iloc[i]:
                        future_df.loc[future_df.index[i], 'Signal'] = 'SELL'
                
                # 进度更新
                self.progress.emit(100)
                
                # 返回预测结果
                self.finished.emit(future_df)
                
            except Exception as e:
                import traceback
                print(traceback.format_exc())
                self.error.emit(f"预测过程中发生错误: {str(e)}")
    
    class IntraDayDataThread(QThread):
        """获取股票15分钟K线数据的线程"""
        finished = pyqtSignal(object)
        error = pyqtSignal(str)
        progress = pyqtSignal(int)
        
        def __init__(self, stock_code, period='7d', interval='15m'):
            super().__init__()
            self.stock_code = stock_code
            self.period = period  # 默认获取7天的15分钟数据
            self.interval = interval  # 15分钟数据
            
        def run(self):
            try:
                # 更新进度
                self.progress.emit(10)
                
                # 从Yahoo Finance获取15分钟K线数据
                stock = yf.Ticker(self.stock_code)
                data = stock.history(period=self.period, interval=self.interval)
                
                if data.empty:
                    self.error.emit(f"无法获取股票 {self.stock_code} 的15分钟K线数据")
                    return
                
                # 更新进度
                self.progress.emit(30)
                
                # 确保索引是日期格式
                data.index = pd.to_datetime(data.index)
                
                # 计算基础技术指标
                # 移动平均线
                data['MA20'] = data['Close'].rolling(window=20).mean()  # 20个15分钟周期的均线
                data['MA60'] = data['Close'].rolling(window=60).mean()  # 60个15分钟周期的均线
                
                # 指数移动平均线 (EMA)
                data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()
                data['EMA26'] = data['Close'].ewm(span=26, adjust=False).mean()
                data['EMA9'] = data['Close'].ewm(span=9, adjust=False).mean()
                
                # 加权移动平均线 (WMA)
                def calculate_wma(series, window):
                    weights = np.arange(1, window+1)
                    return series.rolling(window).apply(lambda x: np.sum(weights*x) / weights.sum(), raw=True)
                
                data['WMA10'] = calculate_wma(data['Close'], 10)
                data['WMA20'] = calculate_wma(data['Close'], 20)
                
                # 真实波动幅度 (ATR)
                data['High-Low'] = data['High'] - data['Low']
                data['High-PrevClose'] = abs(data['High'] - data['Close'].shift(1))
                data['Low-PrevClose'] = abs(data['Low'] - data['Close'].shift(1))
                data['True_Range'] = data[['High-Low', 'High-PrevClose', 'Low-PrevClose']].max(axis=1)
                data['ATR14'] = data['True_Range'].rolling(window=14).mean()
                
                # 清除临时列
                data = data.drop(['High-Low', 'High-PrevClose', 'Low-PrevClose'], axis=1)
                
                # 能量潮指标 (OBV)
                obv = np.zeros(len(data))
                for i in range(1, len(data)):
                    if data['Close'].iloc[i] > data['Close'].iloc[i-1]:
                        obv[i] = obv[i-1] + data['Volume'].iloc[i]
                    elif data['Close'].iloc[i] < data['Close'].iloc[i-1]:
                        obv[i] = obv[i-1] - data['Volume'].iloc[i]
                    else:
                        obv[i] = obv[i-1]
                
                data['OBV'] = obv
                
                # KDJ随机指标
                window = 14
                # 计算最低价的最低点和最高价的最高点
                low_min = data['Low'].rolling(window=window).min()
                high_max = data['High'].rolling(window=window).max()
                
                # 计算RSV值 (Raw Stochastic Value)
                data['RSV'] = 100 * ((data['Close'] - low_min) / (high_max - low_min))
                
                # 计算K值，D值和J值
                data['K'] = data['RSV'].rolling(window=3).mean()  # 或使用ewm: data['RSV'].ewm(alpha=1/3).mean()
                data['D'] = data['K'].rolling(window=3).mean()    # 或使用ewm: data['K'].ewm(alpha=1/3).mean()
                data['J'] = 3 * data['K'] - 2 * data['D']
                
                # 布林带
                window = 20
                data['BB_Middle'] = data['Close'].rolling(window=window).mean()
                data['BB_Std'] = data['Close'].rolling(window=window).std()
                data['BB_Upper'] = data['BB_Middle'] + (data['BB_Std'] * 2)
                data['BB_Lower'] = data['BB_Middle'] - (data['BB_Std'] * 2)
                data['BB_Width'] = (data['BB_Upper'] - data['BB_Lower']) / data['BB_Middle']
                data['BB_Percent'] = (data['Close'] - data['BB_Lower']) / (data['BB_Upper'] - data['BB_Lower'])
                
                # 更新进度
                self.progress.emit(40)
                
                # RSI
                delta = data['Close'].diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                avg_gain = gain.rolling(window=14).mean()
                avg_loss = loss.rolling(window=14).mean()
                rs = avg_gain / avg_loss
                data['RSI'] = 100 - (100 / (1 + rs))
                
                # MACD
                data['MACD'] = data['EMA12'] - data['EMA26']
                data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
                data['MACD_Hist'] = data['MACD'] - data['Signal_Line']
                
                # 计算涨跌目标标签
                data['Price_Change'] = data['Close'].pct_change()
                data['Target'] = (data['Price_Change'].shift(-1) > 0).astype(int)  # 下一个周期是否上涨
                
                # 创建更多特征
                # 价格波动指标
                data['High_Low_Range'] = (data['High'] - data['Low']) / data['Close']
                data['Close_Open_Range'] = abs(data['Close'] - data['Open']) / data['Open']
                
                # 交易量指标
                data['Volume_Change'] = data['Volume'].pct_change()
                data['Volume_MA20'] = data['Volume'].rolling(window=20).mean()
                data['Volume_Ratio'] = data['Volume'] / data['Volume_MA20']
                
                # 短期量比变化
                data['Volume_Ratio_5_20'] = data['Volume'].rolling(window=5).mean() / data['Volume_MA20']
                
                # 价格与均线偏离值
                data['Price_to_MA20'] = data['Close'] / data['MA20'] - 1
                data['Price_to_MA60'] = data['Close'] / data['MA60'] - 1
                data['Price_to_EMA12'] = data['Close'] / data['EMA12'] - 1
                data['Price_to_EMA26'] = data['Close'] / data['EMA26'] - 1
                
                # 均线趋势强度
                data['MA_Trend'] = data['MA20'] / data['MA20'].shift(5) - 1
                data['EMA_Trend'] = data['EMA12'] / data['EMA12'].shift(5) - 1
                
                # 布林带相对位置
                data['BB_Position'] = (data['Close'] - data['BB_Lower']) / (data['BB_Upper'] - data['BB_Lower'])
                
                # 时间特征
                data['Hour'] = data.index.hour
                data['Minute'] = data.index.minute
                data['Day_of_Week'] = data.index.dayofweek
                
                # 增强时间特征
                # 交易时段分类 (美股市场)
                def get_trading_session(hour, minute):
                    if hour < 10 or (hour == 10 and minute < 30):
                        return 0  # 盘前
                    elif hour < 12:
                        return 1  # 早盘
                    elif hour < 14:
                        return 2  # 午盘
                    elif hour < 16:
                        return 3  # 尾盘
                    else:
                        return 4  # 盘后
                
                data['Trading_Session'] = data.apply(lambda x: get_trading_session(x.name.hour, x.name.minute), axis=1)
                
                # 是否为周一或周五
                data['Is_Monday'] = (data.index.dayofweek == 0).astype(int)
                data['Is_Friday'] = (data.index.dayofweek == 4).astype(int)
                
                # 一天中的时间百分比 (0.0 - 1.0)
                data['Day_Progress'] = (data.index.hour * 60 + data.index.minute) / (24 * 60)
                
                # 周内进度 (0.0 - 1.0)
                data['Week_Progress'] = (data.index.dayofweek * 24 * 60 + data.index.hour * 60 + data.index.minute) / (5 * 24 * 60)
                
                # 月份特征 - 季节性影响
                data['Month'] = data.index.month
                
                # 月初/月末指示器
                data['Is_Month_Start'] = data.index.is_month_start.astype(int)
                data['Is_Month_End'] = data.index.is_month_end.astype(int)
                
                # 季度特征
                data['Quarter'] = data.index.quarter
                
                # 季度初/季度末指示器
                data['Is_Quarter_Start'] = data.index.is_quarter_start.astype(int)
                data['Is_Quarter_End'] = data.index.is_quarter_end.astype(int)
                
                # 交易统计
                # 生成交易时段特定的价格变化指标
                for session in range(5):  # 0-4交易时段
                    session_data = data[data['Trading_Session'] == session]
                    if len(session_data) > 0:
                        session_mean = session_data['Price_Change'].mean()
                        session_std = session_data['Price_Change'].std()
                        # 填回原数据框
                        data.loc[data['Trading_Session'] == session, 'Session_Mean_Return'] = session_mean
                        data.loc[data['Trading_Session'] == session, 'Session_Vol'] = session_std
                
                # 周内日特定价格变化
                for day in range(5):  # 0-4 (周一到周五)
                    day_data = data[data['Day_of_Week'] == day]
                    if len(day_data) > 0:
                        day_mean = day_data['Price_Change'].mean()
                        day_std = day_data['Price_Change'].std()
                        # 填回原数据框
                        data.loc[data['Day_of_Week'] == day, 'Day_Mean_Return'] = day_mean
                        data.loc[data['Day_of_Week'] == day, 'Day_Vol'] = day_std
                
                # 价格波动幅度
                data['Volatility_10'] = data['Close'].pct_change().rolling(window=10).std()
                
                # 更新进度
                self.progress.emit(60)
                
                # 时间相关统计特征可能有NaN值，填充
                for col in ['Session_Mean_Return', 'Session_Vol', 'Day_Mean_Return', 'Day_Vol']:
                    if col in data.columns and data[col].isnull().any():
                        # 使用0填充这些统计特征中的NaN
                        data[col] = data[col].fillna(0)
                
                # 删除NaN值
                data = data.dropna()
                
                # 生成买入卖出信号
                self.generate_signals(data)
                
                # 更新进度
                self.progress.emit(100)
                
                # 返回数据
                self.finished.emit(data)
                
            except Exception as e:
                import traceback
                print(traceback.format_exc())
                self.error.emit(str(e))
                
        def generate_signals(self, data):
            """生成买入卖出信号，整合多个技术指标"""
            # 初始化信号列
            data['Signal'] = 'HOLD'
            
            # 1. 基于MACD交叉生成信号
            for i in range(1, len(data)):
                # MACD金叉
                if data['MACD'].iloc[i] > data['Signal_Line'].iloc[i] and \
                   data['MACD'].iloc[i-1] <= data['Signal_Line'].iloc[i-1]:
                    data.loc[data.index[i], 'Signal'] = 'BUY'
                
                # MACD死叉
                elif data['MACD'].iloc[i] < data['Signal_Line'].iloc[i] and \
                     data['MACD'].iloc[i-1] >= data['Signal_Line'].iloc[i-1]:
                    data.loc[data.index[i], 'Signal'] = 'SELL'
            
            # 2. 结合RSI指标
            for i in range(len(data)):
                # RSI超买区，考虑卖出
                if data['RSI'].iloc[i] > 70 and data['Signal'].iloc[i] == 'HOLD':
                    data.loc[data.index[i], 'Signal'] = 'SELL'
                
                # RSI超卖区，考虑买入
                elif data['RSI'].iloc[i] < 30 and data['Signal'].iloc[i] == 'HOLD':
                    data.loc[data.index[i], 'Signal'] = 'BUY'
            
            # 3. 结合KDJ指标
            for i in range(1, len(data)):
                # K线上穿D线，考虑买入
                if data['K'].iloc[i] > data['D'].iloc[i] and \
                   data['K'].iloc[i-1] <= data['D'].iloc[i-1] and \
                   data['Signal'].iloc[i] == 'HOLD':
                    data.loc[data.index[i], 'Signal'] = 'BUY'
                
                # K线下穿D线，考虑卖出
                elif data['K'].iloc[i] < data['D'].iloc[i] and \
                     data['K'].iloc[i-1] >= data['D'].iloc[i-1] and \
                     data['Signal'].iloc[i] == 'HOLD':
                    data.loc[data.index[i], 'Signal'] = 'SELL'
            
            # 4. 布林带突破
            for i in range(len(data)):
                # 价格突破上轨，考虑卖出（可能超买）
                if data['Close'].iloc[i] > data['BB_Upper'].iloc[i] and \
                   data['Signal'].iloc[i] == 'HOLD':
                    data.loc[data.index[i], 'Signal'] = 'SELL'
                
                # 价格突破下轨，考虑买入（可能超卖）
                elif data['Close'].iloc[i] < data['BB_Lower'].iloc[i] and \
                     data['Signal'].iloc[i] == 'HOLD':
                    data.loc[data.index[i], 'Signal'] = 'BUY'
    
    class XGBoostTrainThread(QThread):
        """XGBoost模型训练线程"""
        finished = pyqtSignal(object)
        progress = pyqtSignal(int)
        error = pyqtSignal(str)
        
        def __init__(self, data):
            super().__init__()
            self.data = data
            
        def run(self):
            try:
                # 更新进度
                self.progress.emit(10)
                
                # 准备特征和目标变量
                features = [
                    # 价格特征
                    'Open', 'High', 'Low', 'Close', 'Volume',
                    
                    # 移动平均线
                    'MA20', 'MA60', 'EMA12', 'EMA26', 'EMA9', 'WMA10', 'WMA20',
                    
                    # 技术指标
                    'RSI', 'MACD', 'Signal_Line', 'MACD_Hist', 'ATR14', 'OBV',
                    'K', 'D', 'J',
                    
                    # 布林带
                    'BB_Width', 'BB_Percent', 'BB_Position',
                    
                    # 价格波动指标
                    'High_Low_Range', 'Close_Open_Range', 'Volatility_10',
                    
                    # 交易量指标
                    'Volume_Change', 'Volume_Ratio', 'Volume_Ratio_5_20',
                    
                    # 价格与均线关系
                    'Price_to_MA20', 'Price_to_MA60', 'Price_to_EMA12', 'Price_to_EMA26',
                    'MA_Trend', 'EMA_Trend',
                    
                    # 基础时间特征
                    'Hour', 'Minute', 'Day_of_Week',
                    
                    # 增强时间特征
                    'Trading_Session', 'Is_Monday', 'Is_Friday',
                    'Day_Progress', 'Week_Progress',
                    'Month', 'Is_Month_Start', 'Is_Month_End',
                    'Quarter', 'Is_Quarter_Start', 'Is_Quarter_End',
                    
                    # 时间相关统计特征
                    'Session_Mean_Return', 'Session_Vol',
                    'Day_Mean_Return', 'Day_Vol'
                ]
                
                # 检查所有特征是否都在数据中
                missing_features = [feat for feat in features if feat not in self.data.columns]
                if missing_features:
                    print(f"警告: 以下特征在数据中不存在: {missing_features}")
                    features = [feat for feat in features if feat in self.data.columns]
                
                print(f"使用 {len(features)} 个特征训练XGBoost模型")
                
                X = self.data[features].copy()
                y = self.data['Target'].copy()
                
                # 分割训练集和测试集
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # 更新进度
                self.progress.emit(30)
                
                # 使用XGBoost原生API
                try:
                    # 创建DMatrix
                    dtrain = xgb.DMatrix(X_train, label=y_train)
                    dtest = xgb.DMatrix(X_test, label=y_test)
                    
                    # 定义XGBoost参数
                    params = {
                        'max_depth': 6,  # 增加复杂度以处理更多特征
                        'eta': 0.04,  # 降低学习率以避免过拟合
                        'objective': 'binary:logistic',
                        'eval_metric': 'logloss',
                        'tree_method': 'hist',  # 'gpu_hist' 如果有GPU
                        'subsample': 0.8,
                        'colsample_bytree': 0.8,
                        'min_child_weight': 5,  # 增加以减少过拟合风险
                        'gamma': 0.1,  # 增加以减少过拟合风险
                        'seed': 42
                    }
                    
                    # 定义评估列表
                    evallist = [(dtrain, 'train'), (dtest, 'eval')]
                    
                    # 更新进度
                    self.progress.emit(50)
                    
                    # 训练模型
                    print(f"开始使用纯原生API训练XGBoost模型 - 使用 {len(features)} 个特征")
                    num_round = 500  # 增加训练轮数
                    bst = xgb.train(
                        params, 
                        dtrain, 
                        num_round, 
                        evallist, 
                        early_stopping_rounds=50,  # 增加早停轮数
                        verbose_eval=100  # 每100轮显示一次进度
                    )
                    
                    # 获取特征重要性
                    importance = bst.get_score(importance_type='gain')
                    importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
                    print("\n特征重要性 (Top 10):")
                    for feat, score in importance[:10]:
                        print(f"{feat}: {score}")
                    
                    # 更新进度
                    self.progress.emit(80)
                    
                    # 评估模型
                    y_pred_proba = bst.predict(dtest)
                    y_pred = (y_pred_proba > 0.5).astype(int)
                    
                    # 计算评估指标
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, zero_division=0)
                    recall = recall_score(y_test, y_pred, zero_division=0)
                    f1 = f1_score(y_test, y_pred, zero_division=0)
                    
                    print(f"\n模型评估结果:")
                    print(f"准确率: {accuracy:.4f}")
                    print(f"精确率: {precision:.4f}")
                    print(f"召回率: {recall:.4f}")
                    print(f"F1分数: {f1:.4f}")
                    
                    # 生成最近的预测信号
                    recent_data = self.data.tail(10).copy()
                    recent_X = xgb.DMatrix(recent_data[features])
                    recent_pred_proba = bst.predict(recent_X)
                    
                    # 添加预测概率和生成信号
                    recent_data['Pred_Proba'] = recent_pred_proba
                    recent_data['XGB_Signal'] = 'HOLD'
                    
                    # 基于预测概率设置信号 - 动态阈值
                    # 根据历史预测概率分布计算动态阈值
                    all_proba = bst.predict(xgb.DMatrix(self.data[features]))
                    buy_threshold = np.percentile(all_proba, 70)  # 使用第70百分位作为买入阈值
                    sell_threshold = np.percentile(all_proba, 30)  # 使用第30百分位作为卖出阈值
                    
                    print(f"动态阈值 - 买入: {buy_threshold:.4f}, 卖出: {sell_threshold:.4f}")
                    
                    recent_data.loc[recent_data['Pred_Proba'] >= buy_threshold, 'XGB_Signal'] = 'BUY'
                    recent_data.loc[recent_data['Pred_Proba'] <= sell_threshold, 'XGB_Signal'] = 'SELL'
                    
                    # 创建一个兼容scikit-learn API的模型包装器类以便后续使用
                    class XGBWrapper:
                        def __init__(self, model, features):
                            self.model = model
                            self.features = features
                        
                        def predict(self, X):
                            # 确保只使用训练过的特征
                            X_feat = X[self.features].copy()
                            dmat = xgb.DMatrix(X_feat)
                            proba = self.model.predict(dmat)
                            return (proba > 0.5).astype(int)
                        
                        def predict_proba(self, X):
                            # 确保只使用训练过的特征
                            X_feat = X[self.features].copy()
                            dmat = xgb.DMatrix(X_feat)
                            proba = self.model.predict(dmat)
                            return np.vstack((1-proba, proba)).T
                    
                    # 包装模型，并保存使用的特征列表
                    model_wrapper = XGBWrapper(bst, features)
                    
                    # 返回模型和评估结果
                    result = {
                        'model': model_wrapper,  # 使用包装器
                        'metrics': {
                            'accuracy': accuracy,
                            'precision': precision,
                            'recall': recall,
                            'f1': f1
                        },
                        'recent_predictions': recent_data,
                        'feature_importance': importance[:20],  # 返回前20个重要特征
                        'thresholds': {
                            'buy': buy_threshold,
                            'sell': sell_threshold
                        }
                    }
                    
                    print(result)
                    # 更新进度
                    self.progress.emit(100)
                    
                    # 返回结果
                    self.finished.emit(result)
                    
                except Exception as xgb_error:
                    # 如果原生API出错，打印详细错误
                    import traceback
                    print(f"XGBoost原生API错误: {str(xgb_error)}")
                    print(traceback.format_exc())
                    raise  # 重新抛出异常
                    
            except Exception as e:
                import traceback
                print(traceback.format_exc())
                self.error.emit(f"训练XGBoost模型时出错: {str(e)}")
    
    class SimpleStockApp(QMainWindow):
        def __init__(self):
            super().__init__()
            self.title = "简易股票趋势分析APP"
            self.stock_data = None
            self.lstm_model = None
            self.xgb_model = None
            self.intraday_data = None
            self.init_ui()
        
        def init_ui(self):
            self.setWindowTitle(self.title)
            self.setGeometry(100, 100, 1280, 800)
            
            # 创建中央窗口部件和网格布局
            central_widget = QWidget(self)
            self.setCentralWidget(central_widget)
            
            # 创建整体布局
            main_layout = QVBoxLayout()
            
            # 创建控制面板
            control_panel = QHBoxLayout()
            
            # 股票代码输入
            self.stock_code_label = QLabel("股票代码:")
            self.stock_code_input = QLineEdit()
            self.stock_code_input.setPlaceholderText("输入股票代码 (例如: AAPL)")
            control_panel.addWidget(self.stock_code_label)
            control_panel.addWidget(self.stock_code_input)
            
            # 添加数据区间选择
            self.period_label = QLabel("数据区间:")
            self.period_combo = QComboBox()
            self.period_combo.addItems(["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"])
            self.period_combo.setCurrentText("1y")
            control_panel.addWidget(self.period_label)
            control_panel.addWidget(self.period_combo)
            
            # 添加按钮
            self.load_button = QPushButton("加载数据")
            self.load_button.clicked.connect(self.load_stock_data)
            control_panel.addWidget(self.load_button)
            
            self.predict_button = QPushButton("LSTM预测")
            self.predict_button.clicked.connect(self.predict_trend)
            self.predict_button.setEnabled(False)
            control_panel.addWidget(self.predict_button)
            
            # 添加15分钟数据和XGBoost按钮
            self.load_intraday_button = QPushButton("加载15分钟数据")
            self.load_intraday_button.clicked.connect(self.load_intraday_data)
            control_panel.addWidget(self.load_intraday_button)
            
            self.train_xgboost_button = QPushButton("训练XGBoost")
            self.train_xgboost_button.clicked.connect(self.train_xgboost)
            self.train_xgboost_button.setEnabled(False)
            control_panel.addWidget(self.train_xgboost_button)
            
            # 添加时间分析按钮
            self.time_analysis_button = QPushButton("时间特征分析")
            self.time_analysis_button.clicked.connect(self.plot_time_features)
            self.time_analysis_button.setEnabled(False)
            control_panel.addWidget(self.time_analysis_button)
            
            self.clear_button = QPushButton("清空图表")
            self.clear_button.clicked.connect(self.clear_charts)
            control_panel.addWidget(self.clear_button)
            
            # 将控制面板添加到主布局
            main_layout.addLayout(control_panel)
            
            # 创建工具栏
            self.toolbar = self.addToolBar("图表工具栏")
            
            # 添加图表切换按钮
            self.price_chart_action = self.toolbar.addAction("价格图表")
            self.price_chart_action.triggered.connect(self.plot_intraday_data_with_xgb)
            
            self.time_chart_action = self.toolbar.addAction("时间分析")
            self.time_chart_action.triggered.connect(self.plot_time_features)
            
            if hasattr(self, 'feature_importance_action'):
                self.feature_importance_action = self.toolbar.addAction("特征重要性")
                self.feature_importance_action.triggered.connect(lambda: self.display_feature_importance(self.xgb_feature_importance))
            
            # 创建标签页
            self.tabs = QTabWidget()
            
            # 创建图表页面
            self.chart_tab = QWidget()
            chart_layout = QVBoxLayout()
            
            # 创建图表区域
            self.figure = plt.figure(figsize=(15, 10))
            self.canvas = FigureCanvas(self.figure)
            
            # 添加画布到图表布局
            chart_layout.addWidget(self.canvas)
            
            # 设置图表页面布局
            self.chart_tab.setLayout(chart_layout)
            
            # 添加数据表格页
            self.table_tab = QWidget()
            table_layout = QVBoxLayout()
            
            # 创建数据表格
            self.data_table = QTableWidget()
            self.data_table.setColumnCount(8)  # 增加一列用于XGBoost信号
            self.data_table.setHorizontalHeaderLabels(["日期", "开盘价", "最高价", "最低价", "收盘价", "成交量", "LSTM信号", "XGB信号"])
            self.data_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
            
            # 添加表格到表格布局
            table_layout.addWidget(self.data_table)
            
            # 设置表格页面布局
            self.table_tab.setLayout(table_layout)
            
            # 创建指标页
            self.metrics_tab = QWidget()
            metrics_layout = QVBoxLayout()
            
            # 创建LSTM模型指标区域
            self.lstm_metrics_group = QGroupBox("LSTM模型评估指标")
            lstm_metrics_layout = QFormLayout()
            
            self.lstm_accuracy_label = QLabel("未训练")
            self.lstm_precision_label = QLabel("未训练")
            self.lstm_recall_label = QLabel("未训练")
            self.lstm_f1_label = QLabel("未训练")
            
            lstm_metrics_layout.addRow("准确率:", self.lstm_accuracy_label)
            lstm_metrics_layout.addRow("精确率:", self.lstm_precision_label)
            lstm_metrics_layout.addRow("召回率:", self.lstm_recall_label)
            lstm_metrics_layout.addRow("F1分数:", self.lstm_f1_label)
            
            self.lstm_metrics_group.setLayout(lstm_metrics_layout)
            metrics_layout.addWidget(self.lstm_metrics_group)
            
            # 创建XGBoost模型指标区域
            self.xgb_metrics_group = QGroupBox("XGBoost模型评估指标")
            xgb_metrics_layout = QFormLayout()
            
            self.xgb_accuracy_label = QLabel("未训练")
            self.xgb_precision_label = QLabel("未训练")
            self.xgb_recall_label = QLabel("未训练")
            self.xgb_f1_label = QLabel("未训练")
            
            xgb_metrics_layout.addRow("准确率:", self.xgb_accuracy_label)
            xgb_metrics_layout.addRow("精确率:", self.xgb_precision_label)
            xgb_metrics_layout.addRow("召回率:", self.xgb_recall_label)
            xgb_metrics_layout.addRow("F1分数:", self.xgb_f1_label)
            
            self.xgb_metrics_group.setLayout(xgb_metrics_layout)
            metrics_layout.addWidget(self.xgb_metrics_group)
            
            # 设置指标页面布局
            self.metrics_tab.setLayout(metrics_layout)
            
            # 将页面添加到标签页控件
            self.tabs.addTab(self.chart_tab, "图表")
            self.tabs.addTab(self.table_tab, "数据")
            self.tabs.addTab(self.metrics_tab, "评估指标")
            
            # 添加标签页到主布局
            main_layout.addWidget(self.tabs)
            
            # 创建状态栏
            self.status_bar = QStatusBar()
            self.setStatusBar(self.status_bar)
            self.status_bar.showMessage("就绪")
            
            # 创建进度条
            self.progress_bar = QProgressBar()
            self.progress_bar.setMaximum(100)
            self.progress_bar.setMinimum(0)
            self.progress_bar.setValue(0)
            self.progress_bar.setTextVisible(True)
            self.status_bar.addPermanentWidget(self.progress_bar)
            
            # 设置中央窗口的布局
            central_widget.setLayout(main_layout)
        
        def load_stock_data(self):
            """加载股票日线数据"""
            stock_code = self.stock_code_input.text()
            if not stock_code:
                QMessageBox.warning(self, "警告", "请输入股票代码")
                return
            
            period = self.period_combo.currentText()
            
            # 禁用按钮
            self.load_button.setEnabled(False)
            self.status_bar.showMessage(f"正在加载 {stock_code} 的数据...")
            
            # 创建并启动数据加载线程
            self.data_thread = StockDataThread(stock_code, period)
            self.data_thread.finished.connect(self.on_data_loaded)
            self.data_thread.error.connect(self.on_data_error)
            self.data_thread.progress.connect(self.update_progress)
            self.data_thread.start()
        
        def on_data_loaded(self, data):
            """数据加载完成回调"""
            self.stock_data = data
            self.status_bar.showMessage(f"已加载 {len(data)} 条股票数据")
            self.load_button.setEnabled(True)
            self.predict_button.setEnabled(True)
            
            # 绘制股票K线图
            self.plot_stock_data()
            
            # 更新表格
            self.update_table_with_stock_data()
            
            # 重置进度条
            self.progress_bar.setValue(0)
        
        def on_data_error(self, error_msg):
            """数据加载错误回调"""
            QMessageBox.critical(self, "错误", f"加载股票数据时出错: {error_msg}")
            self.status_bar.showMessage(f"加载股票数据失败: {error_msg}")
            self.load_button.setEnabled(True)
            self.progress_bar.setValue(0)
        
        def plot_stock_data(self):
            """绘制股票K线图表"""
            if self.stock_data is None or self.stock_data.empty:
                return
            
            # 清空图表
            self.figure.clear()
            
            # 创建子图
            ax1 = self.figure.add_subplot(311)  # 价格和均线
            ax2 = self.figure.add_subplot(312)  # MACD
            ax3 = self.figure.add_subplot(313)  # RSI
            
            # 绘制价格和均线
            ax1.plot(self.stock_data.index, self.stock_data['Close'], label='收盘价', color='black')
            ax1.plot(self.stock_data.index, self.stock_data['MA5'], label='MA5', color='blue')
            ax1.plot(self.stock_data.index, self.stock_data['MA20'], label='MA20', color='red')
            
            # 绘制买入卖出信号
            buy_signals = self.stock_data[self.stock_data['Signal'] == 'BUY']
            sell_signals = self.stock_data[self.stock_data['Signal'] == 'SELL']
            
            ax1.scatter(buy_signals.index, buy_signals['Close'], marker='^', color='green', s=100, label='买入信号')
            ax1.scatter(sell_signals.index, sell_signals['Close'], marker='v', color='red', s=100, label='卖出信号')
            
            # 设置图表格式
            ax1.set_title('股票价格与均线交叉信号')
            ax1.set_ylabel('价格')
            ax1.legend()
            ax1.grid(True)
            
            # 绘制MACD
            self.plot_macd(ax2, self.stock_data.index, self.stock_data['MACD'], 
                         self.stock_data['Signal_Line'], self.stock_data['MACD'] - self.stock_data['Signal_Line'])
            
            # 绘制RSI
            ax3.plot(self.stock_data.index, self.stock_data['RSI'], label='RSI', color='purple')
            ax3.axhline(y=70, color='r', linestyle='--', label='超买')
            ax3.axhline(y=30, color='g', linestyle='--', label='超卖')
            ax3.set_title('RSI指标')
            ax3.set_ylabel('RSI值')
            ax3.set_ylim([0, 100])
            ax3.legend()
            ax3.grid(True)
            
            # 调整子图之间的间距
            self.figure.tight_layout()
            
            # 刷新画布
            self.canvas.draw()
        
        def update_table_with_stock_data(self):
            """更新表格数据为股票日线数据"""
            if self.stock_data is None or self.stock_data.empty:
                return
            
            # 清空表格
            self.data_table.setRowCount(0)
            
            # 获取最后30条记录
            display_data = self.stock_data.tail(30)
            
            # 设置表格行数
            self.data_table.setRowCount(len(display_data))
            
            # 填充表格
            for i, (idx, row) in enumerate(display_data.iterrows()):
                # 日期
                date_item = QTableWidgetItem(idx.strftime('%Y-%m-%d'))
                self.data_table.setItem(i, 0, date_item)
                
                # 价格数据
                self.data_table.setItem(i, 1, QTableWidgetItem(f"{row['Open']:.2f}"))
                self.data_table.setItem(i, 2, QTableWidgetItem(f"{row['High']:.2f}"))
                self.data_table.setItem(i, 3, QTableWidgetItem(f"{row['Low']:.2f}"))
                self.data_table.setItem(i, 4, QTableWidgetItem(f"{row['Close']:.2f}"))
                self.data_table.setItem(i, 5, QTableWidgetItem(f"{row['Volume']:.0f}"))
                
                # 信号
                signal_item = QTableWidgetItem(row['Signal'])
                if row['Signal'] == 'BUY':
                    signal_item.setBackground(QColor(144, 238, 144))  # 浅绿色
                elif row['Signal'] == 'SELL':
                    signal_item.setBackground(QColor(255, 182, 193))  # 浅红色
                self.data_table.setItem(i, 6, signal_item)
                
                # XGBoost信号列留空
                self.data_table.setItem(i, 7, QTableWidgetItem("N/A"))
            
            # 调整表格列宽
            self.data_table.resizeColumnsToContents()
        
        def predict_trend(self):
            """使用LSTM预测股价走势"""
            if self.stock_data is None or self.stock_data.empty:
                QMessageBox.warning(self, "警告", "请先加载股票数据")
                return
            
            stock_code = self.stock_code_input.text()
            
            # 禁用按钮
            self.predict_button.setEnabled(False)
            self.status_bar.showMessage(f"正在使用LSTM预测 {stock_code} 的走势...")
            
            # 创建并启动预测线程
            self.prediction_thread = PredictionThread(self.stock_data, stock_code=stock_code)
            self.prediction_thread.finished.connect(self.on_prediction_finished)
            self.prediction_thread.error.connect(self.on_prediction_error)
            self.prediction_thread.progress.connect(self.update_progress)
            self.prediction_thread.start()
        
        def on_prediction_finished(self, prediction_data):
            """预测完成回调"""
            self.prediction_data = prediction_data
            
            # 更新状态
            accuracy_info = ""
            if hasattr(prediction_data, 'attrs') and 'rmse' in prediction_data.attrs:
                rmse = prediction_data.attrs['rmse']
                mape = prediction_data.attrs['mape']
                accuracy_info = f" (RMSE: {rmse:.2f}, MAPE: {mape:.2f}%)"
            
            self.status_bar.showMessage(f"LSTM预测完成{accuracy_info}")
            self.predict_button.setEnabled(True)
            
            # 更新LSTM评估指标
            if hasattr(prediction_data, 'attrs') and 'rmse' in prediction_data.attrs:
                self.update_lstm_metrics(prediction_data.attrs)
            
            # 绘制预测图表
            self.plot_prediction()
            
            # 重置进度条
            self.progress_bar.setValue(0)
        
        def on_prediction_error(self, error_msg):
            """预测错误回调"""
            QMessageBox.critical(self, "错误", f"LSTM预测时出错: {error_msg}")
            self.status_bar.showMessage(f"LSTM预测失败: {error_msg}")
            self.predict_button.setEnabled(True)
            self.progress_bar.setValue(0)
        
        def update_lstm_metrics(self, metrics):
            """更新LSTM模型评估指标"""
            rmse = metrics.get('rmse', 0)
            mape = metrics.get('mape', 0)
            
            # 计算一个近似准确率 (仅用于展示)
            accuracy = max(0, 1 - min(1, mape/100))
            
            self.lstm_accuracy_label.setText(f"{accuracy:.4f}")
            self.lstm_precision_label.setText(f"{1-mape/200:.4f}")
            self.lstm_recall_label.setText(f"{1-rmse/100:.4f}")
            self.lstm_f1_label.setText(f"{1-(rmse+mape/2)/100:.4f}")
        
        def plot_prediction(self):
            """绘制预测结果图表"""
            if self.stock_data is None or not hasattr(self, 'prediction_data') or self.prediction_data is None:
                return
            
            # 清空图表
            self.figure.clear()
            
            # 创建子图
            ax = self.figure.add_subplot(111)
            
            # 绘制历史价格 (最后30个交易日)
            history_data = self.stock_data.tail(30)
            ax.plot(history_data.index, history_data['Close'], 
                   label='历史价格', color='blue', linewidth=2)
            
            # 绘制预测价格
            pred_data = self.prediction_data
            ax.plot(pred_data.index, pred_data['Predicted_Close'], 
                   label='预测价格', color='red', linewidth=2, linestyle='--')
            
            # 绘制交易信号
            for i, idx in enumerate(pred_data.index):
                signal = pred_data['Signal'].iloc[i]
                price = pred_data['Predicted_Close'].iloc[i]
                
                if signal == 'BUY':
                    ax.scatter(idx, price, marker='^', color='green', s=120, zorder=5, label='_买入')
                elif signal == 'SELL':
                    ax.scatter(idx, price, marker='v', color='red', s=120, zorder=5, label='_卖出')
            
            # 添加垂直线标记历史数据与预测数据的分界
            ax.axvline(x=history_data.index[-1], color='black', linestyle='--', alpha=0.7)
            ax.text(history_data.index[-1], ax.get_ylim()[0], '今日',
                   horizontalalignment='center', verticalalignment='bottom', rotation=90)
            
            # 添加预测区域背景
            ax.axvspan(history_data.index[-1], pred_data.index[-1], alpha=0.1, color='yellow')
            
            # 设置图表格式
            ax.set_title('LSTM价格预测结果')
            ax.set_xlabel('日期')
            ax.set_ylabel('价格')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.5)
            
            # 格式化x轴日期
            self.figure.autofmt_xdate()
            
            # 调整布局并绘制
            self.figure.tight_layout()
            self.canvas.draw()
        
        def plot_macd(self, ax, dates, macd, signal, histogram):
            """绘制MACD图表"""
            # 绘制MACD和信号线
            ax.plot(dates, macd, label='MACD', color='blue', linewidth=1.5)
            ax.plot(dates, signal, label='信号线', color='red', linewidth=1.5)
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # 绘制直方图
            for i in range(len(dates)):
                if i < len(histogram):
                    color = 'green' if histogram[i] > 0 else 'red'
                    ax.bar(dates[i], histogram[i], color=color, width=0.7, alpha=0.6)
            
            # 设置标题和标签
            ax.set_title('MACD指标')
            ax.set_ylabel('值')
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.5)
        
        def load_intraday_data(self):
            """加载15分钟K线数据"""
            stock_code = self.stock_code_input.text()
            if not stock_code:
                QMessageBox.warning(self, "警告", "请输入股票代码")
                return
            
            # 禁用按钮
            self.load_intraday_button.setEnabled(False)
            self.status_bar.showMessage(f"正在加载 {stock_code} 的15分钟K线数据...")
            
            # 创建并启动15分钟数据加载线程
            self.intraday_thread = IntraDayDataThread(stock_code)
            self.intraday_thread.finished.connect(self.on_intraday_data_loaded)
            self.intraday_thread.error.connect(self.on_intraday_error)
            self.intraday_thread.progress.connect(self.update_progress)
            self.intraday_thread.start()
        
        def on_intraday_data_loaded(self, data):
            """15分钟K线数据加载完成回调"""
            self.intraday_data = data
            self.status_bar.showMessage(f"已加载 {len(data)} 条15分钟K线数据")
            self.load_intraday_button.setEnabled(True)
            self.train_xgboost_button.setEnabled(True)
            self.time_analysis_button.setEnabled(True)
            
            # 绘制15分钟K线图
            self.plot_intraday_data()
            
            # 更新表格
            self.update_table_with_intraday_data()
            
            # 重置进度条
            self.progress_bar.setValue(0)
        
        def on_intraday_error(self, error_msg):
            """15分钟K线数据加载错误回调"""
            QMessageBox.critical(self, "错误", f"加载15分钟K线数据时出错: {error_msg}")
            self.status_bar.showMessage(f"加载15分钟K线数据失败: {error_msg}")
            self.load_intraday_button.setEnabled(True)
            self.progress_bar.setValue(0)
        
        def plot_intraday_data(self):
            """绘制15分钟K线图表"""
            if self.intraday_data is None or self.intraday_data.empty:
                return
            
            # 清空图表
            self.figure.clear()
            
            # 创建子图
            ax1 = self.figure.add_subplot(311)  # 价格和均线
            ax2 = self.figure.add_subplot(312)  # MACD
            ax3 = self.figure.add_subplot(313)  # RSI
            
            # 绘制价格和均线
            ax1.plot(self.intraday_data.index, self.intraday_data['Close'], label='收盘价', color='black')
            ax1.plot(self.intraday_data.index, self.intraday_data['MA20'], label='MA20', color='blue')
            ax1.plot(self.intraday_data.index, self.intraday_data['MA60'], label='MA60', color='red')
            
            # 绘制买入卖出信号
            buy_signals = self.intraday_data[self.intraday_data['Signal'] == 'BUY']
            sell_signals = self.intraday_data[self.intraday_data['Signal'] == 'SELL']
            
            ax1.scatter(buy_signals.index, buy_signals['Close'], marker='^', color='green', s=100, label='买入信号')
            ax1.scatter(sell_signals.index, sell_signals['Close'], marker='v', color='red', s=100, label='卖出信号')
            
            # 设置图表格式
            ax1.set_title('15分钟K线价格与信号')
            ax1.set_ylabel('价格')
            ax1.legend()
            ax1.grid(True)
            
            # 绘制MACD
            self.plot_macd(ax2, self.intraday_data.index, self.intraday_data['MACD'], 
                         self.intraday_data['Signal_Line'], self.intraday_data['MACD_Hist'])
            
            # 绘制RSI
            ax3.plot(self.intraday_data.index, self.intraday_data['RSI'], label='RSI', color='purple')
            ax3.axhline(y=70, color='r', linestyle='--', label='超买')
            ax3.axhline(y=30, color='g', linestyle='--', label='超卖')
            ax3.set_title('RSI指标')
            ax3.set_ylabel('RSI值')
            ax3.set_ylim([0, 100])
            ax3.legend()
            ax3.grid(True)
            
            # 调整子图之间的间距
            self.figure.tight_layout()
            
            # 刷新画布
            self.canvas.draw()
        
        def update_table_with_intraday_data(self):
            """更新表格数据为15分钟K线数据"""
            if self.intraday_data is None or self.intraday_data.empty:
                return
            
            # 清空表格
            self.data_table.setRowCount(0)
            
            # 获取最后30条记录
            display_data = self.intraday_data.tail(30)
            
            # 设置表格行数
            self.data_table.setRowCount(len(display_data))
            
            # 填充表格
            for i, (idx, row) in enumerate(display_data.iterrows()):
                # 日期
                date_item = QTableWidgetItem(idx.strftime('%Y-%m-%d %H:%M'))
                self.data_table.setItem(i, 0, date_item)
                
                # 价格数据
                self.data_table.setItem(i, 1, QTableWidgetItem(f"{row['Open']:.2f}"))
                self.data_table.setItem(i, 2, QTableWidgetItem(f"{row['High']:.2f}"))
                self.data_table.setItem(i, 3, QTableWidgetItem(f"{row['Low']:.2f}"))
                self.data_table.setItem(i, 4, QTableWidgetItem(f"{row['Close']:.2f}"))
                self.data_table.setItem(i, 5, QTableWidgetItem(f"{row['Volume']:.0f}"))
                
                # 信号
                signal_item = QTableWidgetItem(row['Signal'])
                if row['Signal'] == 'BUY':
                    signal_item.setBackground(QColor(144, 238, 144))  # 浅绿色
                elif row['Signal'] == 'SELL':
                    signal_item.setBackground(QColor(255, 182, 193))  # 浅红色
                self.data_table.setItem(i, 6, signal_item)
                
                # XGBoost信号（如果有）
                if 'XGB_Signal' in row:
                    xgb_signal_item = QTableWidgetItem(row['XGB_Signal'])
                    if row['XGB_Signal'] == 'BUY':
                        xgb_signal_item.setBackground(QColor(144, 238, 144))  # 浅绿色
                    elif row['XGB_Signal'] == 'SELL':
                        xgb_signal_item.setBackground(QColor(255, 182, 193))  # 浅红色
                    self.data_table.setItem(i, 7, xgb_signal_item)
                else:
                    self.data_table.setItem(i, 7, QTableWidgetItem("N/A"))
            
            # 调整表格列宽
            self.data_table.resizeColumnsToContents()
        
        def train_xgboost(self):
            """训练XGBoost模型"""
            if self.intraday_data is None or self.intraday_data.empty:
                QMessageBox.warning(self, "警告", "请先加载15分钟K线数据")
                return
            
            # 禁用按钮
            self.train_xgboost_button.setEnabled(False)
            self.status_bar.showMessage("正在训练XGBoost模型...")
            
            # 创建并启动XGBoost训练线程
            self.xgb_thread = XGBoostTrainThread(self.intraday_data)
            self.xgb_thread.finished.connect(self.on_xgb_model_trained)
            self.xgb_thread.error.connect(self.on_xgb_error)
            self.xgb_thread.progress.connect(self.update_progress)
            self.xgb_thread.start()
        
        def on_xgb_model_trained(self, result):
            """XGBoost模型训练完成回调"""
            self.xgb_model = result['model']
            metrics = result['metrics']
            recent_predictions = result['recent_predictions']
            
            # 更新状态
            self.status_bar.showMessage("XGBoost模型训练完成")
            self.train_xgboost_button.setEnabled(True)
            
            # 更新模型评估指标
            self.update_xgb_metrics(metrics)
            
            # 保存特征重要性（如果有）
            if 'feature_importance' in result:
                self.xgb_feature_importance = result['feature_importance']
                self.display_feature_importance(result['feature_importance'])
                
                # 添加特征重要性按钮到工具栏，如果尚未添加
                if not hasattr(self, 'feature_importance_action') or self.feature_importance_action is None:
                    self.feature_importance_action = self.toolbar.addAction("特征重要性")
                    self.feature_importance_action.triggered.connect(lambda: self.display_feature_importance(self.xgb_feature_importance))
            
            # 更新最近预测数据
            self.intraday_data.update(recent_predictions)
            
            # 确保时间分析按钮可用
            self.time_analysis_button.setEnabled(True)
            
            # 更新图表和表格
            self.plot_intraday_data_with_xgb()
            self.update_table_with_intraday_data()
            
            # 重置进度条
            self.progress_bar.setValue(0)

        def display_feature_importance(self, feature_importance):
            """显示特征重要性"""
            # 清空图表
            self.figure.clear()
            
            # 创建子图
            ax = self.figure.add_subplot(111)
            
            # 提取特征名称和得分
            features = [item[0] for item in feature_importance]
            scores = [item[1] for item in feature_importance]
            
            # 创建水平条形图
            y_pos = np.arange(len(features))
            ax.barh(y_pos, scores, align='center')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(features)
            ax.invert_yaxis()  # 使最高的特征在顶部
            ax.set_xlabel('重要性得分')
            ax.set_title('XGBoost特征重要性')
            
            # 调整布局
            self.figure.tight_layout()
            
            # 刷新画布
            self.canvas.draw()
            
            # 延迟切换回数据图表
            QTimer.singleShot(5000, self.plot_intraday_data_with_xgb)

        def plot_intraday_data_with_xgb(self):
            """绘制15分钟K线图表，包含XGBoost预测信号"""
            if self.intraday_data is None or self.intraday_data.empty:
                return
            
            # 清空图表
            self.figure.clear()
            
            # 创建一个更高的图像以容纳所有图表
            self.figure.set_size_inches(15, 14)  # 增加高度
            
            # 创建GridSpec对象进行更灵活的布局
            gs = self.figure.add_gridspec(4, 1, height_ratios=[2, 1, 1, 1], hspace=0.3)
            
            # 创建子图
            ax1 = self.figure.add_subplot(gs[0, 0])  # 价格和均线（占据更多空间）
            ax2 = self.figure.add_subplot(gs[1, 0])  # KDJ指标
            ax3 = self.figure.add_subplot(gs[2, 0])  # MACD
            ax4 = self.figure.add_subplot(gs[3, 0])  # RSI
            
            # 设置图表标题
            self.figure.suptitle(f'股票代码: {self.stock_code_input.text()} - 15分钟K线技术分析', fontsize=16)
            
            # 绘制价格和均线、布林带
            ax1.plot(self.intraday_data.index, self.intraday_data['Close'], label='收盘价', color='black', linewidth=1.5)
            ax1.plot(self.intraday_data.index, self.intraday_data['MA20'], label='MA20', color='blue', linewidth=1.2)
            ax1.plot(self.intraday_data.index, self.intraday_data['EMA12'], label='EMA12', color='green', linewidth=1.2)
            
            # 绘制布林带（如果存在）
            if all(col in self.intraday_data.columns for col in ['BB_Upper', 'BB_Lower']):
                ax1.plot(self.intraday_data.index, self.intraday_data['BB_Upper'], 
                        label='布林上轨', color='red', linestyle='--', alpha=0.6)
                ax1.plot(self.intraday_data.index, self.intraday_data['BB_Lower'], 
                        label='布林下轨', color='green', linestyle='--', alpha=0.6)
                ax1.fill_between(self.intraday_data.index, 
                                self.intraday_data['BB_Upper'], 
                                self.intraday_data['BB_Lower'], 
                                alpha=0.1, color='gray')
            
            # 绘制基础信号
            buy_signals = self.intraday_data[self.intraday_data['Signal'] == 'BUY']
            sell_signals = self.intraday_data[self.intraday_data['Signal'] == 'SELL']
            
            ax1.scatter(buy_signals.index, buy_signals['Close'], marker='^', color='green', s=90, alpha=0.8, label='技术指标买入')
            ax1.scatter(sell_signals.index, sell_signals['Close'], marker='v', color='red', s=90, alpha=0.8, label='技术指标卖出')
            
            # 绘制XGBoost信号
            if 'XGB_Signal' in self.intraday_data.columns:
                xgb_buy = self.intraday_data[self.intraday_data['XGB_Signal'] == 'BUY']
                xgb_sell = self.intraday_data[self.intraday_data['XGB_Signal'] == 'SELL']
                
                ax1.scatter(xgb_buy.index, xgb_buy['Close'], marker='*', color='green', s=150, label='XGBoost买入')
                ax1.scatter(xgb_sell.index, xgb_sell['Close'], marker='x', color='red', s=150, label='XGBoost卖出')
            
            # 设置图表格式
            ax1.set_title('价格与信号', fontsize=14)
            ax1.set_ylabel('价格', fontsize=12)
            ax1.legend(loc='upper left', fontsize='small', ncol=2)  # 使用两列显示图例
            ax1.grid(True, alpha=0.3)
            
            # 优化x轴日期格式
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
            ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
            
            # 绘制KDJ
            if all(col in self.intraday_data.columns for col in ['K', 'D', 'J']):
                ax2.plot(self.intraday_data.index, self.intraday_data['K'], label='K', color='blue', linewidth=1.2)
                ax2.plot(self.intraday_data.index, self.intraday_data['D'], label='D', color='orange', linewidth=1.2)
                ax2.plot(self.intraday_data.index, self.intraday_data['J'], label='J', color='purple', linewidth=1)
                ax2.axhline(y=80, color='r', linestyle='--', linewidth=0.8, label='超买')
                ax2.axhline(y=20, color='g', linestyle='--', linewidth=0.8, label='超卖')
                ax2.set_title('KDJ指标', fontsize=14)
                ax2.set_ylabel('值', fontsize=12)
                ax2.legend(loc='upper left', fontsize='small', ncol=3)
                ax2.grid(True, alpha=0.3)
                ax2.set_ylim([0, 120])  # 固定KDJ图表的Y轴范围
                
                # 同步x轴
                ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
                ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
            
            # 绘制MACD - 使用更清晰的表示方式
            self.plot_macd_enhanced(ax3, self.intraday_data)
            
            # 绘制RSI
            ax4.plot(self.intraday_data.index, self.intraday_data['RSI'], label='RSI', color='purple', linewidth=1.5)
            ax4.axhline(y=70, color='r', linestyle='--', linewidth=0.8, label='超买')
            ax4.axhline(y=30, color='g', linestyle='--', linewidth=0.8, label='超卖')
            ax4.set_title('RSI指标', fontsize=14)
            ax4.set_ylabel('RSI值', fontsize=12)
            ax4.set_ylim([0, 100])
            ax4.legend(loc='upper left', fontsize='small')
            ax4.grid(True, alpha=0.3)
            
            # 同步x轴
            ax4.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
            ax4.xaxis.set_major_locator(mdates.AutoDateLocator())
            
            # 格式化所有子图的x轴为日期，并设置一致的刻度
            for ax in [ax1, ax2, ax3, ax4]:
                plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
            
            # 恢复原始图像尺寸
            plt.tight_layout()
            
            # 刷新画布
            self.canvas.draw()
            
        def plot_macd_enhanced(self, ax, data):
            """绘制增强版MACD图表"""
            dates = data.index
            macd = data['MACD']
            signal = data['Signal_Line']
            histogram = data['MACD_Hist']
            
            # 绘制MACD和信号线
            ax.plot(dates, macd, label='MACD', color='blue', linewidth=1.5)
            ax.plot(dates, signal, label='信号线', color='red', linewidth=1.5)
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.8)
            
            # 创建柱状图
            pos_hist = histogram.copy()
            neg_hist = histogram.copy()
            pos_hist[pos_hist <= 0] = np.nan
            neg_hist[neg_hist > 0] = np.nan
            
            # 绘制正值和负值柱状图，使用不同颜色
            ax.bar(dates, pos_hist, color='green', width=0.6, alpha=0.6, label='正柱')
            ax.bar(dates, neg_hist, color='red', width=0.6, alpha=0.6, label='负柱')
            
            # 标记金叉和死叉
            for i in range(1, len(dates)):
                # 金叉 - MACD从下方穿过信号线
                if macd.iloc[i] > signal.iloc[i] and macd.iloc[i-1] <= signal.iloc[i-1]:
                    ax.scatter(dates[i], macd.iloc[i], s=80, marker='^', color='green', zorder=5)
                
                # 死叉 - MACD从上方穿过信号线
                elif macd.iloc[i] < signal.iloc[i] and macd.iloc[i-1] >= signal.iloc[i-1]:
                    ax.scatter(dates[i], macd.iloc[i], s=80, marker='v', color='red', zorder=5)
            
            # 设置标题和标签
            ax.set_title('MACD指标', fontsize=14)
            ax.set_ylabel('值', fontsize=12)
            ax.legend(loc='upper left', fontsize='small', ncol=4)
            ax.grid(True, alpha=0.3)
            
            # 设置x轴格式
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        
        def update_progress(self, value):
            """更新进度条"""
            self.progress_bar.setValue(value)
        
        def clear_charts(self):
            """清空所有图表和数据"""
            # 清空数据
            self.stock_data = None
            self.intraday_data = None
            if hasattr(self, 'prediction_data'):
                self.prediction_data = None
            self.xgb_model = None
            self.lstm_model = None
            
            # 清空特征重要性
            if hasattr(self, 'xgb_feature_importance'):
                self.xgb_feature_importance = None
            
            # 清空图表
            self.figure.clear()
            self.canvas.draw()
            
            # 清空表格
            self.data_table.setRowCount(0)
            
            # 重置指标标签
            self.lstm_accuracy_label.setText("未训练")
            self.lstm_precision_label.setText("未训练")
            self.lstm_recall_label.setText("未训练")
            self.lstm_f1_label.setText("未训练")
            self.xgb_accuracy_label.setText("未训练")
            self.xgb_precision_label.setText("未训练")
            self.xgb_recall_label.setText("未训练")
            self.xgb_f1_label.setText("未训练")
            
            # 禁用按钮
            self.predict_button.setEnabled(False)
            self.train_xgboost_button.setEnabled(False)
            self.time_analysis_button.setEnabled(False)
            
            # 更新状态
            self.status_bar.showMessage("已清空所有图表和数据")
            self.progress_bar.setValue(0)

        def on_xgb_error(self, error_msg):
            """XGBoost模型训练错误回调"""
            QMessageBox.critical(self, "错误", f"训练XGBoost模型时出错: {error_msg}")
            self.status_bar.showMessage(f"训练XGBoost模型失败: {error_msg}")
            self.train_xgboost_button.setEnabled(True)
            self.progress_bar.setValue(0)

        def update_xgb_metrics(self, metrics):
            """更新XGBoost模型评估指标"""
            self.xgb_accuracy_label.setText(f"{metrics['accuracy']:.4f}")
            self.xgb_precision_label.setText(f"{metrics['precision']:.4f}")
            self.xgb_recall_label.setText(f"{metrics['recall']:.4f}")
            self.xgb_f1_label.setText(f"{metrics['f1']:.4f}")
    
        def plot_time_features(self):
            """绘制时间特征分析图表"""
            if self.intraday_data is None or self.intraday_data.empty:
                return
            
            # 清空图表
            self.figure.clear()
            
            # 根据数据中是否有时间统计特征决定图表布局
            has_time_stats = all(col in self.intraday_data.columns for col in ['Day_Mean_Return', 'Session_Mean_Return'])
            
            if has_time_stats:
                # 创建子图
                ax1 = self.figure.add_subplot(221)  # 周内回报分析
                ax2 = self.figure.add_subplot(222)  # 交易时段回报分析
                ax3 = self.figure.add_subplot(223)  # 小时回报分析
                ax4 = self.figure.add_subplot(224)  # 周内波动率分析
                
                # 绘制周内回报分析
                day_returns = self.intraday_data.groupby('Day_of_Week')['Price_Change'].mean() * 100  # 转换为百分比
                days = ['周一', '周二', '周三', '周四', '周五']
                colors = ['green' if x >= 0 else 'red' for x in day_returns]
                ax1.bar(days, day_returns, color=colors)
                ax1.set_title('周内平均回报率')
                ax1.set_ylabel('平均回报率 (%)')
                for i, v in enumerate(day_returns):
                    ax1.text(i, v + 0.01 if v >= 0 else v - 0.1, f"{v:.2f}%", ha='center')
                
                # 绘制交易时段回报分析
                if 'Trading_Session' in self.intraday_data.columns:
                    session_returns = self.intraday_data.groupby('Trading_Session')['Price_Change'].mean() * 100
                    sessions = ['盘前', '早盘', '午盘', '尾盘', '盘后']
                    colors = ['green' if x >= 0 else 'red' for x in session_returns]
                    ax2.bar(sessions[:len(session_returns)], session_returns, color=colors)
                    ax2.set_title('交易时段平均回报率')
                    ax2.set_ylabel('平均回报率 (%)')
                    for i, v in enumerate(session_returns):
                        ax2.text(i, v + 0.01 if v >= 0 else v - 0.1, f"{v:.2f}%", ha='center')
                
                # 绘制小时回报分析
                hour_returns = self.intraday_data.groupby('Hour')['Price_Change'].mean() * 100
                hours = hour_returns.index
                colors = ['green' if x >= 0 else 'red' for x in hour_returns]
                ax3.bar(hours, hour_returns, color=colors)
                ax3.set_title('小时平均回报率')
                ax3.set_xlabel('小时')
                ax3.set_ylabel('平均回报率 (%)')
                
                # 绘制周内波动率分析
                day_vol = self.intraday_data.groupby('Day_of_Week')['Price_Change'].std() * 100
                ax4.bar(days, day_vol, color='blue', alpha=0.7)
                ax4.set_title('周内波动率')
                ax4.set_ylabel('波动率 (%)')
                for i, v in enumerate(day_vol):
                    ax4.text(i, v + 0.01, f"{v:.2f}%", ha='center')
            else:
                # 创建子图 - 简化版
                ax1 = self.figure.add_subplot(221)  # 周内分析
                ax2 = self.figure.add_subplot(222)  # 小时分析
                ax3 = self.figure.add_subplot(212)  # 交易量分析
                
                # 周内交易量分析
                day_volume = self.intraday_data.groupby('Day_of_Week')['Volume'].mean()
                days = ['周一', '周二', '周三', '周四', '周五']
                ax1.bar(days, day_volume, color='blue')
                ax1.set_title('周内平均交易量')
                ax1.set_ylabel('交易量')
                
                # 小时交易量分析
                hour_volume = self.intraday_data.groupby('Hour')['Volume'].mean()
                hours = hour_volume.index
                ax2.bar(hours, hour_volume, color='green')
                ax2.set_title('小时平均交易量')
                ax2.set_xlabel('小时')
                ax2.set_ylabel('交易量')
                
                # 分钟K线收盘价与交易量关系
                ax3.scatter(self.intraday_data['Volume'], self.intraday_data['Close'], alpha=0.5)
                ax3.set_title('交易量与价格关系')
                ax3.set_xlabel('交易量')
                ax3.set_ylabel('收盘价')
            
            # 调整子图之间的间距
            self.figure.tight_layout()
            
            # 刷新画布
            self.canvas.draw()
    
    def check_dependencies():
        """检查所需的库是否安装"""
        missing_deps = []
        
        try:
            import yfinance
        except ImportError:
            missing_deps.append("yfinance")
        
        try:
            import tensorflow
        except ImportError:
            missing_deps.append("tensorflow")
        
        try:
            import sklearn
        except ImportError:
            missing_deps.append("scikit-learn")
        
        if missing_deps:
            print(f"错误: 缺少必要的库: {', '.join(missing_deps)}")
            print("\n请安装必要的库：")
            print("python3 -m pip install PyQt5 matplotlib pandas numpy yfinance tensorflow scikit-learn")
            return False
        
        return True
    
    def main():
        # 检查依赖
        if not check_dependencies():
            return
        
        app = QApplication(sys.argv)
        window = SimpleStockApp()
        window.show()
        
        sys.exit(app.exec_())
    
    if __name__ == "__main__":
        main()
        
except ImportError as e:
    print("错误: 缺少必要的库")
    print(f"详细错误: {e}")
    print("\n请安装必要的库：")
    print("python3 -m pip install PyQt5 matplotlib pandas numpy yfinance tensorflow scikit-learn")
    sys.exit(1) 