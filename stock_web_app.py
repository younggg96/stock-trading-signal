#!/usr/bin/env python3
"""
股票交易信号可视化 - 网页版应用
基于Plotly和Dash构建，提供交互式图表和实时更新
"""
import os
import pandas as pd
import numpy as np
import datetime
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import warnings
warnings.filterwarnings('ignore')

# 导入数据处理功能从原应用程序
try:
    from qt_stock_app import IntraDayDataThread
except ImportError:
    # 为Vercel部署定义一个占位符类
    class IntraDayDataThread:
        pass

# 配置
DEFAULT_STOCK = "AAPL"
DEFAULT_PERIOD = "7d"
DEFAULT_INTERVAL = "15m"

# 创建assets文件夹并添加CSS文件
if not os.path.exists('assets'):
    os.makedirs('assets')

# 创建CSS文件
css_path = os.path.join('assets', 'custom.css')
if not os.path.exists(css_path):
    with open(css_path, 'w') as f:
        f.write('''
/* 下拉框样式 */
.dash-dropdown .Select-control {
    background-color: #2c3e50 !important;
    color: white !important;
    border: 1px solid #3498db !important;
    border-radius: 6px !important;
}

.dash-dropdown .Select-menu-outer {
    background-color: #2c3e50 !important;
    color: white !important;
    border: 1px solid #3498db !important;
    border-radius: 0 0 6px 6px !important;
}

.dash-dropdown .Select-value-label {
    color: white !important;
}

.dash-dropdown .Select-menu {
    background-color: #2c3e50 !important;
}

.dash-dropdown .Select-option {
    background-color: #2c3e50 !important;
    color: white !important;
}

.dash-dropdown .Select-option:hover {
    background-color: #34495e !important;
}

.dash-dropdown .Select-option.is-focused {
    background-color: #34495e !important;
}

.dash-dropdown .Select-option.is-selected {
    background-color: #3498db !important;
}

.dash-dropdown .Select-placeholder {
    color: rgba(255, 255, 255, 0.8) !important;
}

.dash-dropdown .Select-arrow {
    border-color: white transparent transparent !important;
}
''')

# 初始化应用
app = dash.Dash(
    __name__, 
    external_stylesheets=[dbc.themes.DARKLY],
    title="股票交易信号可视化",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)

# 为Vercel部署设置服务器
server = app.server

# 页面布局
app.layout = dbc.Container([
    # 使用dcc.Store替代html.Style和html.Div
    
    dbc.Row([
        dbc.Col([
            html.H2("股票交易信号分析", className="text-center my-4"),
            html.Hr(),
        ], width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("设置"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("股票代码:"),
                            dbc.Input(id="stock-input", type="text", value=DEFAULT_STOCK, placeholder="例如: AAPL"),
                        ], width=4),
                        
                        dbc.Col([
                            html.Label("数据区间:"),
                            dcc.Dropdown(
                                id="period-dropdown",
                                options=[
                                    {"label": "1天", "value": "1d"},
                                    {"label": "7天", "value": "7d"},
                                    {"label": "1个月", "value": "1mo"},
                                    {"label": "3个月", "value": "3mo"},
                                ],
                                value=DEFAULT_PERIOD,
                                className="dash-dropdown"
                            ),
                        ], width=4),
                        
                        dbc.Col([
                            html.Label("K线间隔:"),
                            dcc.Dropdown(
                                id="interval-dropdown",
                                options=[
                                    {"label": "1分钟", "value": "1m"},
                                    {"label": "5分钟", "value": "5m"},
                                    {"label": "15分钟", "value": "15m"},
                                    {"label": "30分钟", "value": "30m"},
                                    {"label": "1小时", "value": "1h"},
                                ],
                                value=DEFAULT_INTERVAL,
                                className="dash-dropdown"
                            ),
                        ], width=4),
                    ]),
                    
                    dbc.Row([
                        dbc.Col([
                            html.Label("趋势线设置:"),
                            dbc.Row([
                                dbc.Col([
                                    html.Label("窗口大小:"),
                                    dcc.Slider(
                                        id="trend-window-slider",
                                        min=3,
                                        max=20,
                                        step=1,
                                        value=10,
                                        marks={i: str(i) for i in range(3, 21, 2)},
                                    ),
                                ], width=6),
                                dbc.Col([
                                    html.Label("斜率阈值:"),
                                    dcc.Slider(
                                        id="trend-slope-slider",
                                        min=0.0001,
                                        max=0.01,
                                        step=0.0001,
                                        value=0.001,
                                        marks={i/10000: str(i/10000) for i in range(1, 101, 20)},
                                    ),
                                ], width=6),
                            ]),
                        ], width=12),
                    ], className="mt-2"),
                    
                    dbc.Row([
                        dbc.Col([
                            dbc.Button("加载数据", id="load-button", color="primary", className="mt-3 w-100"),
                        ], width=12),
                    ]),
                    
                    dbc.Row([
                        dbc.Col([
                            dbc.Spinner(html.Div(id="loading-output", className="mt-2")),
                        ], width=12),
                    ]),
                ]),
            ], className="mb-4"),
        ], width=12),
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Tabs([
                dbc.Tab(label="技术指标图表", tab_id="tab-charts", children=[
                    dcc.Graph(id="price-chart", style={"height": "350px"}, config={"displayModeBar": True}),
                    dcc.Graph(id="kdj-chart", style={"height": "250px"}, config={"displayModeBar": False}),
                    dcc.Graph(id="macd-chart", style={"height": "250px"}, config={"displayModeBar": False}),
                    dcc.Graph(id="rsi-chart", style={"height": "250px"}, config={"displayModeBar": False}),
                ]),
                dbc.Tab(label="时间特征分析", tab_id="tab-time", children=[
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("选择时间特征"),
                                dbc.CardBody([
                                    dcc.Dropdown(
                                        id="time-feature-dropdown",
                                        options=[
                                            {"label": "小时效应分析", "value": "hour"},
                                            {"label": "星期效应分析", "value": "day_of_week"}
                                        ],
                                        value=None,
                                        placeholder="请选择时间特征分析类型",
                                        className="dash-dropdown mb-3"
                                    )
                                ])
                            ], className="mb-3")
                        ], width=12)
                    ]),
                    dcc.Graph(id="time-feature-chart", style={"height": "600px"}, config={"displayModeBar": True}),
                ]),
                dbc.Tab(label="数据表格", tab_id="tab-table", children=[
                    html.Div(id="table-container", className="mt-3")
                ]),
            ], id="tabs", active_tab="tab-charts"),
        ], width=12),
    ]),
    
    dcc.Store(id="stock-data"),
    dcc.Store(id="trend-lines-data"),
    
], fluid=True)

# 数据加载回调
@app.callback(
    [
        Output("stock-data", "data"),
        Output("loading-output", "children"),
    ],
    [Input("load-button", "n_clicks")],
    [
        State("stock-input", "value"),
        State("period-dropdown", "value"),
        State("interval-dropdown", "value"),
    ],
    prevent_initial_call=True
)
def load_stock_data(n_clicks, stock_code, period, interval):
    if not n_clicks:
        return None, ""
    
    if not stock_code:
        return None, html.Div("请输入股票代码", className="text-danger")
    
    try:
        # 从Yahoo Finance获取15分钟K线数据
        stock = yf.Ticker(stock_code)
        data = stock.history(period=period, interval=interval)
        
        if data.empty:
            return None, html.Div(f"无法获取股票 {stock_code} 的K线数据", className="text-danger")
        
        # 确保索引是日期格式
        data.index = pd.to_datetime(data.index)
        
        # 计算技术指标，代码从qt_stock_app.py中提取，简化版
        # 移动平均线
        data['MA20'] = data['Close'].rolling(window=20).mean()
        data['MA60'] = data['Close'].rolling(window=60).mean()
        
        # 指数移动平均线
        data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()
        data['EMA26'] = data['Close'].ewm(span=26, adjust=False).mean()
        data['EMA9'] = data['Close'].ewm(span=9, adjust=False).mean()
        
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
        
        # KDJ
        window = 14
        low_min = data['Low'].rolling(window=window).min()
        high_max = data['High'].rolling(window=window).max()
        data['RSV'] = 100 * ((data['Close'] - low_min) / (high_max - low_min))
        data['K'] = data['RSV'].rolling(window=3).mean()
        data['D'] = data['K'].rolling(window=3).mean()
        data['J'] = 3 * data['K'] - 2 * data['D']
        
        # 布林带
        window = 20
        data['BB_Middle'] = data['Close'].rolling(window=window).mean()
        data['BB_Std'] = data['Close'].rolling(window=window).std()
        data['BB_Upper'] = data['BB_Middle'] + (data['BB_Std'] * 2)
        data['BB_Lower'] = data['BB_Middle'] - (data['BB_Std'] * 2)
        
        # 生成买入卖出信号
        data['Signal'] = 'HOLD'
        
        # 基于MACD交叉生成信号
        for i in range(1, len(data)):
            # MACD金叉
            if data['MACD'].iloc[i] > data['Signal_Line'].iloc[i] and \
               data['MACD'].iloc[i-1] <= data['Signal_Line'].iloc[i-1]:
                data.loc[data.index[i], 'Signal'] = 'BUY'
            
            # MACD死叉
            elif data['MACD'].iloc[i] < data['Signal_Line'].iloc[i] and \
                 data['MACD'].iloc[i-1] >= data['Signal_Line'].iloc[i-1]:
                data.loc[data.index[i], 'Signal'] = 'SELL'
        
        # 结合RSI指标
        for i in range(len(data)):
            # RSI超买区，考虑卖出
            if data['RSI'].iloc[i] > 70 and data['Signal'].iloc[i] == 'HOLD':
                data.loc[data.index[i], 'Signal'] = 'SELL'
            
            # RSI超卖区，考虑买入
            elif data['RSI'].iloc[i] < 30 and data['Signal'].iloc[i] == 'HOLD':
                data.loc[data.index[i], 'Signal'] = 'BUY'
        
        # 增加时间特征
        data['Hour'] = data.index.hour
        data['Minute'] = data.index.minute
        data['Day_of_Week'] = data.index.dayofweek
        
        # 添加星期名称映射
        day_mapping = {0: '周一', 1: '周二', 2: '周三', 3: '周四', 4: '周五', 5: '周六', 6: '周日'}
        data['Day_Name'] = data['Day_of_Week'].map(day_mapping)
        
        # 删除NaN值
        data = data.fillna(method='ffill').fillna(method='bfill')
        
        # 重置索引，确保Date列存在于数据中而不仅是索引
        data = data.reset_index()
        
        # 确保索引列名为'Date'而不是'Datetime'
        if 'Datetime' in data.columns:
            data = data.rename(columns={'Datetime': 'Date'})
            
        # 将数据转换为JSON格式以便存储 - 使用date_format='iso'确保日期正确转换
        data_json = data.to_json(date_format='iso', orient='split')
        
        return data_json, html.Div(f"已成功加载 {stock_code} 的 {len(data)} 条{interval}K线数据", className="text-success")
    
    except Exception as e:
        import traceback
        print(f"加载数据时出错: {str(e)}")
        print(traceback.format_exc())
        return None, html.Div(f"加载数据时出错: {str(e)}", className="text-danger")

# 图表更新回调
@app.callback(
    [
        Output("price-chart", "figure"),
        Output("kdj-chart", "figure"),
        Output("macd-chart", "figure"),
        Output("rsi-chart", "figure"),
    ],
    [Input("stock-data", "data"), Input("trend-lines-data", "data")],
    prevent_initial_call=True
)
def update_charts(json_data, trend_lines_data):
    if not json_data:
        # 返回空图表
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="无数据",
            xaxis=dict(title="请加载数据"),
            yaxis=dict(title=""),
            template="plotly_dark"
        )
        return empty_fig, empty_fig, empty_fig, empty_fig
    
    try:
        # 解析数据
        try:
            data = pd.read_json(json_data, orient='split')
            print(f"成功解析JSON数据，数据形状: {data.shape}，列: {data.columns.tolist()}")
        except Exception as e:
            print(f"JSON解析失败: {str(e)}")
            error_fig = go.Figure()
            error_fig.add_annotation(
                text=f"JSON解析失败: {str(e)}",
                x=0.5, y=0.5,
                xref="paper", yref="paper",
                showarrow=False,
                font=dict(color="red", size=14)
            )
            error_fig.update_layout(title="数据格式错误", template="plotly_dark")
            return error_fig, error_fig, error_fig, error_fig
        
        # 处理日期列，兼容不同的命名
        if 'Datetime' in data.columns and 'Date' not in data.columns:
            data = data.rename(columns={'Datetime': 'Date'})
            print("将'Datetime'列重命名为'Date'")
        
        # 确保Date列是日期时间类型并设置为索引
        try:
            if 'Date' not in data.columns:
                print("数据中缺少Date列")
                error_fig = go.Figure()
                error_fig.add_annotation(
                    text="数据中缺少Date列",
                    x=0.5, y=0.5,
                    xref="paper", yref="paper",
                    showarrow=False,
                    font=dict(color="red", size=14)
                )
                error_fig.update_layout(title="数据格式错误", template="plotly_dark")
                return error_fig, error_fig, error_fig, error_fig
            
            data['Date'] = pd.to_datetime(data['Date'])
            data.set_index('Date', inplace=True)
            print("成功将Date列设置为索引")
        except Exception as e:
            print(f"设置日期索引失败: {str(e)}")
            error_fig = go.Figure()
            error_fig.add_annotation(
                text=f"设置日期索引失败: {str(e)}",
                x=0.5, y=0.5,
                xref="paper", yref="paper",
                showarrow=False,
                font=dict(color="red", size=14)
            )
            error_fig.update_layout(title="日期数据错误", template="plotly_dark")
            return error_fig, error_fig, error_fig, error_fig
        
        # 验证必要的列是否存在
        required_cols = ['Close']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            error_fig = go.Figure()
            error_fig.add_annotation(
                text=f"数据中缺少必要的列: {', '.join(missing_cols)}",
                x=0.5, y=0.5,
                xref="paper", yref="paper",
                showarrow=False,
                font=dict(color="red", size=14)
            )
            error_fig.update_layout(title="数据格式错误", template="plotly_dark")
            return error_fig, error_fig, error_fig, error_fig
        
        # 提取买入卖出信号点
        buy_signals = data[data['Signal'] == 'BUY'] if 'Signal' in data.columns else pd.DataFrame()
        sell_signals = data[data['Signal'] == 'SELL'] if 'Signal' in data.columns else pd.DataFrame()
        
        # 创建价格图表
        price_fig = go.Figure()
        
        # 添加K线图
        if all(col in data.columns for col in ['Open', 'High', 'Low', 'Close']):
            price_fig.add_trace(
                go.Candlestick(
                    x=data.index,
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'],
                    name="价格",
                )
            )
        else:
            # 如果缺少K线所需的列，只绘制收盘价
            if 'Close' in data.columns:
                price_fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['Close'],
                        name="收盘价",
                        line=dict(color='white', width=2)
                    )
                )
            else:
                # 没有价格数据时显示错误
                price_fig.add_annotation(
                    text="没有价格数据",
                    x=0.5, y=0.5,
                    xref="paper", yref="paper",
                    showarrow=False,
                    font=dict(color="red", size=14)
                )
        
        # 检查均线列是否存在
        if 'MA20' in data.columns:
            price_fig.add_trace(go.Scatter(x=data.index, y=data['MA20'], name='MA20', line=dict(color='blue', width=1)))
        
        if 'EMA12' in data.columns:
            price_fig.add_trace(go.Scatter(x=data.index, y=data['EMA12'], name='EMA12', line=dict(color='green', width=1)))
        
        # 检查布林带列是否存在
        if all(col in data.columns for col in ['BB_Upper', 'BB_Lower']):
            price_fig.add_trace(go.Scatter(
                x=data.index, y=data['BB_Upper'], 
                name='布林上轨', 
                line=dict(color='red', width=1, dash='dash'),
                opacity=0.7
            ))
            
            price_fig.add_trace(go.Scatter(
                x=data.index, y=data['BB_Lower'], 
                name='布林下轨', 
                line=dict(color='green', width=1, dash='dash'),
                opacity=0.7,
                fill='tonexty', 
                fillcolor='rgba(200,200,200,0.1)'
            ))
        
        # 添加买入卖出信号
        if not buy_signals.empty and 'Close' in buy_signals.columns:
            price_fig.add_trace(go.Scatter(
                x=buy_signals.index,
                y=buy_signals['Close'],
                mode='markers',
                name='买入信号',
                marker=dict(
                    color='green',
                    size=10,
                    symbol='triangle-up',
                    line=dict(color='green', width=1)
                )
            ))
        
        if not sell_signals.empty and 'Close' in sell_signals.columns:
            price_fig.add_trace(go.Scatter(
                x=sell_signals.index,
                y=sell_signals['Close'],
                mode='markers',
                name='卖出信号',
                marker=dict(
                    color='red',
                    size=10,
                    symbol='triangle-down',
                    line=dict(color='red', width=1)
                )
            ))
            
        # 添加趋势线
        try:
            if trend_lines_data:
                # 从存储的JSON数据中提取趋势线
                support_lines_json = trend_lines_data.get('support_lines', [])
                resistance_lines_json = trend_lines_data.get('resistance_lines', [])
                last_index_str = trend_lines_data.get('last_index')
                last_index = pd.to_datetime(last_index_str) if last_index_str else None
                
                # 添加支撑线(最多显示5条最显著的线)
                significant_support_lines = support_lines_json[:min(5, len(support_lines_json))]
                for i, line_data in enumerate(significant_support_lines):
                    x1 = pd.to_datetime(line_data['x1'])
                    y1 = line_data['y1']
                    x2 = pd.to_datetime(line_data['x2'])
                    y2 = line_data['y2']
                    
                    price_fig.add_shape(
                        type="line",
                        x0=x1, y0=y1, x1=x2, y1=y2,
                        line=dict(
                            color="green",
                            width=2,
                            dash="solid",
                        ),
                        name=f"支撑线{i+1}"
                    )
                    
                    # 向右扩展趋势线
                    if last_index and x2 < last_index:
                        # 计算斜率
                        slope = (y2 - y1) / (x2 - x1).total_seconds()
                        # 计算扩展时间
                        time_extend = (last_index - x2).total_seconds()
                        # 计算扩展点的y值
                        y_extend = y2 + slope * time_extend
                        # 添加扩展线
                        price_fig.add_shape(
                            type="line",
                            x0=x2, y0=y2, x1=last_index, y1=y_extend,
                            line=dict(
                                color="green",
                                width=2,
                                dash="dot",
                            )
                        )
                
                # 添加阻力线(最多显示5条最显著的线)
                significant_resistance_lines = resistance_lines_json[:min(5, len(resistance_lines_json))]
                for i, line_data in enumerate(significant_resistance_lines):
                    x1 = pd.to_datetime(line_data['x1'])
                    y1 = line_data['y1']
                    x2 = pd.to_datetime(line_data['x2'])
                    y2 = line_data['y2']
                    
                    price_fig.add_shape(
                        type="line",
                        x0=x1, y0=y1, x1=x2, y1=y2,
                        line=dict(
                            color="red",
                            width=2,
                            dash="solid",
                        ),
                        name=f"阻力线{i+1}"
                    )
                    
                    # 向右扩展趋势线
                    if last_index and x2 < last_index:
                        # 计算斜率
                        slope = (y2 - y1) / (x2 - x1).total_seconds()
                        # 计算扩展时间
                        time_extend = (last_index - x2).total_seconds()
                        # 计算扩展点的y值
                        y_extend = y2 + slope * time_extend
                        # 添加扩展线
                        price_fig.add_shape(
                            type="line",
                            x0=x2, y0=y2, x1=last_index, y1=y_extend,
                            line=dict(
                                color="red",
                                width=2,
                                dash="dot",
                            )
                        )
                
                # 添加趋势线数量标签
                price_fig.add_annotation(
                    text=f"支撑线: {len(significant_support_lines)}  阻力线: {len(significant_resistance_lines)}",
                    align="left",
                    x=0.01,
                    y=0.99,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(color="white", size=10),
                )
            else:
                price_fig.add_annotation(
                    text="未找到趋势线数据",
                    align="left",
                    x=0.01,
                    y=0.99,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(color="yellow", size=10),
                )
                
        except Exception as e:
            print(f"添加趋势线时出错: {str(e)}")
            # 添加错误标签
            price_fig.add_annotation(
                text=f"趋势线绘制失败: {str(e)}",
                align="left",
                x=0.01,
                y=0.99,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(color="red", size=10),
            )
        
        # 设置价格图表布局
        if len(data.index) > 0:
            start_date = data.index[0].strftime('%Y-%m-%d') if hasattr(data.index[0], 'strftime') else "未知日期"
            end_date = data.index[-1].strftime('%Y-%m-%d') if hasattr(data.index[-1], 'strftime') else "未知日期"
            title = f"{start_date} 到 {end_date} 价格走势"
        else:
            title = "价格走势"
            
        price_fig.update_layout(
            title=title,
            xaxis_title="日期",
            yaxis_title="价格",
            xaxis_rangeslider_visible=False,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=50, r=50, t=80, b=50),
            template="plotly_dark",
        )
        
        # 创建KDJ图表
        kdj_fig = go.Figure()
        
        # 检查KDJ列是否存在
        if all(col in data.columns for col in ['K', 'D', 'J']):
            # 添加KDJ线
            kdj_fig.add_trace(go.Scatter(x=data.index, y=data['K'], name='K', line=dict(color='blue', width=1.5)))
            kdj_fig.add_trace(go.Scatter(x=data.index, y=data['D'], name='D', line=dict(color='orange', width=1.5)))
            kdj_fig.add_trace(go.Scatter(x=data.index, y=data['J'], name='J', line=dict(color='purple', width=1)))
            
            # 添加超买超卖线
            if len(data.index) > 0:
                kdj_fig.add_shape(
                    type="line", line=dict(color="red", width=1, dash="dash"),
                    y0=80, y1=80, x0=data.index[0], x1=data.index[-1]
                )
                
                kdj_fig.add_shape(
                    type="line", line=dict(color="green", width=1, dash="dash"),
                    y0=20, y1=20, x0=data.index[0], x1=data.index[-1]
                )
        else:
            # 如果KDJ数据不存在，添加提示
            kdj_fig.add_annotation(
                text="KDJ数据不可用",
                x=0.5, y=0.5,
                xref="paper", yref="paper",
                showarrow=False,
                font=dict(color="white", size=14)
            )
        
        # 设置KDJ图表布局
        kdj_fig.update_layout(
            title="KDJ指标",
            xaxis_title="",
            yaxis_title="值",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=50, r=50, t=60, b=30),
            template="plotly_dark",
            yaxis=dict(range=[0, 120]),
        )
        
        # 创建MACD图表
        macd_fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # 检查MACD列是否存在
        if all(col in data.columns for col in ['MACD', 'Signal_Line', 'MACD_Hist']):
            # 添加MACD和信号线
            macd_fig.add_trace(
                go.Scatter(x=data.index, y=data['MACD'], name='MACD', line=dict(color='blue', width=1.5)),
                secondary_y=False,
            )
            
            macd_fig.add_trace(
                go.Scatter(x=data.index, y=data['Signal_Line'], name='信号线', line=dict(color='red', width=1.5)),
                secondary_y=False,
            )
            
            # 添加MACD柱状图
            colors = ['green' if val >= 0 else 'red' for val in data['MACD_Hist']]
            
            macd_fig.add_trace(
                go.Bar(
                    x=data.index, 
                    y=data['MACD_Hist'], 
                    name='MACD直方图',
                    marker_color=colors,
                    opacity=0.7
                ),
                secondary_y=True,
            )
            
            # 标记金叉和死叉
            macd_cross_buy = []
            macd_cross_sell = []
            
            for i in range(1, len(data)):
                # 金叉
                if data['MACD'].iloc[i] > data['Signal_Line'].iloc[i] and \
                   data['MACD'].iloc[i-1] <= data['Signal_Line'].iloc[i-1]:
                    macd_cross_buy.append(i)
                # 死叉
                elif data['MACD'].iloc[i] < data['Signal_Line'].iloc[i] and \
                     data['MACD'].iloc[i-1] >= data['Signal_Line'].iloc[i-1]:
                    macd_cross_sell.append(i)
            
            if macd_cross_buy and len(macd_cross_buy) > 0:
                macd_fig.add_trace(
                    go.Scatter(
                        x=data.iloc[macd_cross_buy].index,
                        y=data['MACD'].iloc[macd_cross_buy],
                        mode='markers',
                        name='MACD金叉',
                        marker=dict(
                            color='green',
                            size=10,
                            symbol='triangle-up',
                            line=dict(color='green', width=1)
                        )
                    ),
                    secondary_y=False,
                )
            
            if macd_cross_sell and len(macd_cross_sell) > 0:
                macd_fig.add_trace(
                    go.Scatter(
                        x=data.iloc[macd_cross_sell].index,
                        y=data['MACD'].iloc[macd_cross_sell],
                        mode='markers',
                        name='MACD死叉',
                        marker=dict(
                            color='red',
                            size=10,
                            symbol='triangle-down',
                            line=dict(color='red', width=1)
                        )
                    ),
                    secondary_y=False,
                )
        else:
            # 如果MACD数据不存在，添加提示
            macd_fig.add_annotation(
                text="MACD数据不可用",
                x=0.5, y=0.5,
                xref="paper", yref="paper",
                showarrow=False,
                font=dict(color="white", size=14)
            )
        
        # 设置MACD图表布局
        macd_fig.update_layout(
            title="MACD指标",
            xaxis_title="",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=50, r=50, t=60, b=30),
            template="plotly_dark",
        )
        
        macd_fig.update_yaxes(title_text="MACD值", secondary_y=False)
        macd_fig.update_yaxes(title_text="柱状图", showgrid=False, secondary_y=True)
        
        # 创建RSI图表
        rsi_fig = go.Figure()
        
        # 检查RSI列是否存在
        if 'RSI' in data.columns:
            # 添加RSI线
            rsi_fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], name='RSI', line=dict(color='purple', width=1.5)))
            
            # 添加超买超卖线
            if len(data.index) > 0:
                rsi_fig.add_shape(
                    type="line", line=dict(color="red", width=1, dash="dash"),
                    y0=70, y1=70, x0=data.index[0], x1=data.index[-1]
                )
                
                rsi_fig.add_shape(
                    type="line", line=dict(color="green", width=1, dash="dash"),
                    y0=30, y1=30, x0=data.index[0], x1=data.index[-1]
                )
        else:
            # 如果RSI数据不存在，添加提示
            rsi_fig.add_annotation(
                text="RSI数据不可用",
                x=0.5, y=0.5,
                xref="paper", yref="paper",
                showarrow=False,
                font=dict(color="white", size=14)
            )
        
        # 设置RSI图表布局
        rsi_fig.update_layout(
            title="RSI指标",
            xaxis_title="日期",
            yaxis_title="RSI值",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=50, r=50, t=60, b=50),
            template="plotly_dark",
            yaxis=dict(range=[0, 100]),
        )
        
        # 统一所有图表的X轴范围
        if len(data.index) > 0:
            x_range = [data.index[0], data.index[-1]]
            price_fig.update_xaxes(range=x_range)
            kdj_fig.update_xaxes(range=x_range)
            macd_fig.update_xaxes(range=x_range)
            rsi_fig.update_xaxes(range=x_range)
        
        return price_fig, kdj_fig, macd_fig, rsi_fig
    
    except Exception as e:
        import traceback
        print(f"图表生成错误: {str(e)}")
        print(traceback.format_exc())
        
        # 返回错误图表
        error_fig = go.Figure()
        error_fig.add_annotation(
            text=f"图表生成错误: {str(e)}",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(color="red", size=14)
        )
        error_fig.update_layout(title="数据处理错误", template="plotly_dark")
        
        return error_fig, error_fig, error_fig, error_fig

# 时间特征图表回调
@app.callback(
    Output("time-feature-chart", "figure"),
    [Input("stock-data", "data"), Input("time-feature-dropdown", "value")],
    prevent_initial_call=True
)
def update_time_chart(json_data, feature):
    if not json_data or not feature:
        # 返回空图表
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="请选择时间特征",
            xaxis=dict(title="无数据"),
            yaxis=dict(title=""),
            template="plotly_dark"
        )
        return empty_fig
    
    try:
        # 解析数据
        data = pd.read_json(json_data, orient='split')
        print(f"时间特征分析 - 解析JSON数据，形状: {data.shape}，列: {data.columns.tolist()}")
        
        # 处理日期列，兼容不同的命名
        if 'Datetime' in data.columns and 'Date' not in data.columns:
            data = data.rename(columns={'Datetime': 'Date'})
            print("时间特征分析 - 将'Datetime'列重命名为'Date'")
        
        # 确保Date列是日期时间类型
        if 'Date' in data.columns:
            data['Date'] = pd.to_datetime(data['Date'])
        elif 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'])
            # 统一列名
            data.rename(columns={'date': 'Date'}, inplace=True)
        else:
            # 尝试将索引转为日期类型并创建Date列
            try:
                data.index = pd.to_datetime(data.index)
                data['Date'] = data.index
            except Exception as e:
                print(f"处理日期数据失败: {str(e)}")
                error_fig = go.Figure()
                error_fig.add_annotation(
                    text=f"无法识别日期数据: {str(e)}",
                    x=0.5, y=0.5,
                    xref="paper", yref="paper",
                    showarrow=False,
                    font=dict(color="red", size=14)
                )
                error_fig.update_layout(title="数据格式错误", template="plotly_dark")
                return error_fig
        
        # 确保Close列存在，用于计算价格变化
        if 'Close' not in data.columns:
            error_fig = go.Figure()
            error_fig.add_annotation(
                text="数据中缺少Close列，无法计算价格变化",
                x=0.5, y=0.5,
                xref="paper", yref="paper",
                showarrow=False,
                font=dict(color="red", size=14)
            )
            error_fig.update_layout(title="数据不完整", template="plotly_dark")
            return error_fig
        
        # 创建或确保Hour和Day_of_Week列存在
        if 'Hour' not in data.columns and 'Date' in data.columns:
            data['Hour'] = data['Date'].dt.hour
        
        # 强制重新计算Day_of_Week和Day_Name列
        if 'Date' in data.columns:
            data['Day_of_Week'] = data['Date'].dt.dayofweek
            # 将数字映射到星期名称
            day_mapping = {0: '周一', 1: '周二', 2: '周三', 3: '周四', 4: '周五', 5: '周六', 6: '周日'}
            data['Day_Name'] = data['Day_of_Week'].map(day_mapping)
        
        # 计算价格变化百分比
        data['Price_Change'] = data['Close'].pct_change() * 100
        
        # 创建图表
        fig = go.Figure()
        
        if feature == 'hour':
            if 'Hour' not in data.columns:
                fig.add_annotation(
                    text="Hour列不存在，无法分析小时特征",
                    x=0.5, y=0.5,
                    xref="paper", yref="paper",
                    showarrow=False,
                    font=dict(color="red", size=14)
                )
                fig.update_layout(title="数据不完整", template="plotly_dark")
                return fig
                
            # 分析小时特征
            hourly_analysis = data.groupby('Hour')['Price_Change'].agg(['mean', 'std', 'count']).reset_index()
            
            if hourly_analysis.empty or hourly_analysis['count'].sum() == 0:
                fig.add_annotation(
                    text="没有足够的数据进行小时分析",
                    x=0.5, y=0.5,
                    xref="paper", yref="paper",
                    showarrow=False,
                    font=dict(color="white", size=14)
                )
                fig.update_layout(title="数据不足", template="plotly_dark")
                return fig
            
            # 添加小时平均价格变化柱状图
            bar_colors = ['green' if x >= 0 else 'red' for x in hourly_analysis['mean']]
            
            fig.add_trace(go.Bar(
                x=hourly_analysis['Hour'],
                y=hourly_analysis['mean'],
                name='平均价格变化(%)',
                marker_color=bar_colors,
                text=hourly_analysis['mean'].round(2),
                textposition='outside',
                error_y=dict(
                    type='data',
                    array=hourly_analysis['std'],
                    visible=True
                )
            ))
            
            # 添加交易数量线图
            fig.add_trace(go.Scatter(
                x=hourly_analysis['Hour'],
                y=hourly_analysis['count'],
                name='交易数量',
                yaxis='y2',
                line=dict(color='yellow', width=2)
            ))
            
            # 设置小时特征图表布局
            fig.update_layout(
                title='交易时段分析 - 小时效应',
                xaxis=dict(
                    title='小时',
                    tickvals=list(range(24)),
                    ticktext=[f"{h}:00" for h in range(24)]
                ),
                yaxis=dict(
                    title='平均价格变化百分比(%)',
                    side='left'
                ),
                yaxis2=dict(
                    title='交易数量',
                    side='right',
                    overlaying='y',
                    showgrid=False
                ),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                template="plotly_dark"
            )
        
        elif feature == 'day_of_week':
            # 检查必要的列是否存在
            if 'Day_of_Week' not in data.columns:
                fig.add_annotation(
                    text="Day_of_Week列不存在，无法分析星期特征",
                    x=0.5, y=0.5,
                    xref="paper", yref="paper",
                    showarrow=False,
                    font=dict(color="red", size=14)
                )
                fig.update_layout(title="数据不完整", template="plotly_dark")
                return fig
                
            if 'Day_Name' not in data.columns:
                # 尝试重新创建Day_Name列
                try:
                    day_mapping = {0: '周一', 1: '周二', 2: '周三', 3: '周四', 4: '周五', 5: '周六', 6: '周日'}
                    data['Day_Name'] = data['Day_of_Week'].map(day_mapping)
                except Exception as e:
                    fig.add_annotation(
                        text=f"无法创建Day_Name列: {str(e)}",
                        x=0.5, y=0.5,
                        xref="paper", yref="paper",
                        showarrow=False,
                        font=dict(color="red", size=14)
                    )
                    fig.update_layout(title="数据处理错误", template="plotly_dark")
                    return fig
            
            # 分析星期特征
            try:
                daily_analysis = data.groupby(['Day_of_Week', 'Day_Name'])['Price_Change'].agg(['mean', 'std', 'count']).reset_index()
                
                if daily_analysis.empty or daily_analysis['count'].sum() == 0:
                    fig.add_annotation(
                        text="没有足够的数据进行星期分析",
                        x=0.5, y=0.5,
                        xref="paper", yref="paper",
                        showarrow=False,
                        font=dict(color="white", size=14)
                    )
                    fig.update_layout(title="数据不足", template="plotly_dark")
                    return fig
                
                # 排序，确保星期一到星期日的顺序
                daily_analysis = daily_analysis.sort_values('Day_of_Week')
                
                # 添加星期平均价格变化柱状图
                bar_colors = ['green' if x >= 0 else 'red' for x in daily_analysis['mean']]
                
                fig.add_trace(go.Bar(
                    x=daily_analysis['Day_Name'],
                    y=daily_analysis['mean'],
                    name='平均价格变化(%)',
                    marker_color=bar_colors,
                    text=daily_analysis['mean'].round(2),
                    textposition='outside',
                    error_y=dict(
                        type='data',
                        array=daily_analysis['std'],
                        visible=True
                    )
                ))
                
                # 添加交易数量线图
                fig.add_trace(go.Scatter(
                    x=daily_analysis['Day_Name'],
                    y=daily_analysis['count'],
                    name='交易数量',
                    yaxis='y2',
                    line=dict(color='yellow', width=2)
                ))
                
                # 设置星期特征图表布局
                fig.update_layout(
                    title='交易时段分析 - 星期效应',
                    xaxis=dict(
                        title='星期',
                        categoryorder='array',
                        categoryarray=daily_analysis['Day_Name'].tolist()
                    ),
                    yaxis=dict(
                        title='平均价格变化百分比(%)',
                        side='left'
                    ),
                    yaxis2=dict(
                        title='交易数量',
                        side='right',
                        overlaying='y',
                        showgrid=False
                    ),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ),
                    template="plotly_dark"
                )
            except Exception as e:
                fig.add_annotation(
                    text=f"星期分析处理错误: {str(e)}",
                    x=0.5, y=0.5,
                    xref="paper", yref="paper",
                    showarrow=False,
                    font=dict(color="red", size=14)
                )
                fig.update_layout(title="数据处理错误", template="plotly_dark")
                return fig
        
        else:
            # 未知特征
            fig.add_annotation(
                text=f"未支持的时间特征: {feature}",
                x=0.5, y=0.5,
                xref="paper", yref="paper",
                showarrow=False,
                font=dict(color="white", size=14)
            )
            fig.update_layout(title="未知特征", template="plotly_dark")
        
        return fig
    
    except Exception as e:
        import traceback
        print(f"时间特征分析错误: {str(e)}")
        print(traceback.format_exc())
        
        # 返回错误图表
        error_fig = go.Figure()
        error_fig.add_annotation(
            text=f"时间特征分析错误: {str(e)}",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(color="red", size=14)
        )
        error_fig.update_layout(title="数据处理错误", template="plotly_dark")
        
        return error_fig

# 数据表格回调
@app.callback(
    Output("table-container", "children"),
    [Input("stock-data", "data")],
    prevent_initial_call=True
)
def update_table(json_data):
    if not json_data:
        return html.Div("无数据")
    
    try:
        # 解析数据
        data = pd.read_json(json_data, orient='split')
        print(f"表格更新 - 解析JSON数据，形状: {data.shape}，列: {data.columns.tolist()}")
        
        # 处理日期列，兼容不同的命名
        if 'Datetime' in data.columns and 'Date' not in data.columns:
            data = data.rename(columns={'Datetime': 'Date'})
            print("表格更新 - 将'Datetime'列重命名为'Date'")
        
        # 确保Date列存在并且是日期时间类型
        if 'Date' not in data.columns:
            return html.Div("数据格式错误: 'Date'列不存在", className="text-danger")
        
        data['Date'] = pd.to_datetime(data['Date'])
        
        # 选择显示的列
        display_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Signal']
        display_cols = [col for col in display_cols if col in data.columns]
        
        # 如果没有足够的列来显示，返回错误消息
        if len(display_cols) < 2:  # 至少需要Date列和一个数据列
            return html.Div("表格无足够数据列可显示", className="text-danger")
            
        display_data = data[display_cols].copy()
        
        # 格式化日期
        display_data['Date'] = display_data['Date'].dt.strftime('%Y-%m-%d %H:%M')
        
        # 格式化数值
        numeric_cols = [col for col in ['Open', 'High', 'Low', 'Close'] if col in display_data.columns]
        for col in numeric_cols:
            display_data[col] = display_data[col].round(2)
        
        # 处理交易量列，确保其为数值型
        if 'Volume' in display_data.columns:
            display_data['Volume'] = pd.to_numeric(display_data['Volume'], errors='coerce')
            display_data['Volume'] = display_data['Volume'].fillna(0).round(0).astype(int)
        
        # 创建表格
        table = dbc.Table.from_dataframe(
            display_data.tail(20),  # 只显示最新的20条数据
            striped=True,
            bordered=True,
            hover=True,
            responsive=True,
            color="dark",
        )
        
        return table
    except Exception as e:
        import traceback
        print(f"表格更新错误: {str(e)}")
        print(traceback.format_exc())
        return html.Div(f"表格更新错误: {str(e)}", className="text-danger")

def identify_trend_lines(df, window_size=5, slope_threshold=0.01):
    """识别数据中的趋势线
    
    Args:
        df: 包含价格数据的DataFrame
        window_size: 用于识别局部最高点和最低点的窗口大小
        slope_threshold: 斜率阈值，用于确定线的重要性
        
    Returns:
        元组 (上升趋势线, 下降趋势线)，每个都是一个列表，包含(x1, y1, x2, y2)的点对
    """
    # 确保索引是递增的数字，便于处理
    x = np.arange(len(df))
    y = df['Close'].values
    
    # 查找局部极大值和极小值
    max_idx = []
    min_idx = []
    
    for i in range(window_size, len(df) - window_size):
        # 检查是否是局部最大值
        if y[i] == max(y[i-window_size:i+window_size+1]):
            max_idx.append(i)
        # 检查是否是局部最小值
        if y[i] == min(y[i-window_size:i+window_size+1]):
            min_idx.append(i)
    
    # 找出支撑线（连接低点的线）
    support_lines = []
    for i in range(len(min_idx)):
        for j in range(i+1, len(min_idx)):
            idx1, idx2 = min_idx[i], min_idx[j]
            # 计算斜率
            if idx2 - idx1 > 0:  # 避免除以零
                slope = (y[idx2] - y[idx1]) / (idx2 - idx1)
                # 如果斜率在阈值范围内，添加线
                if abs(slope) > slope_threshold:
                    support_lines.append((
                        x[idx1], y[idx1], x[idx2], y[idx2]
                    ))
    
    # 找出阻力线（连接高点的线）
    resistance_lines = []
    for i in range(len(max_idx)):
        for j in range(i+1, len(max_idx)):
            idx1, idx2 = max_idx[i], max_idx[j]
            # 计算斜率
            if idx2 - idx1 > 0:  # 避免除以零
                slope = (y[idx2] - y[idx1]) / (idx2 - idx1)
                # 如果斜率在阈值范围内，添加线
                if abs(slope) > slope_threshold:
                    resistance_lines.append((
                        x[idx1], y[idx1], x[idx2], y[idx2]
                    ))
    
    # 将索引转换回日期
    result_support = []
    result_resistance = []
    
    for x1, y1, x2, y2 in support_lines:
        result_support.append((
            df.index[int(x1)], y1, df.index[int(x2)], y2
        ))
    
    for x1, y1, x2, y2 in resistance_lines:
        result_resistance.append((
            df.index[int(x1)], y1, df.index[int(x2)], y2
        ))
    
    return result_support, result_resistance

# 趋势线数据计算回调
@app.callback(
    Output("trend-lines-data", "data"),
    [Input("stock-data", "data"), Input("trend-window-slider", "value"), Input("trend-slope-slider", "value")],
    prevent_initial_call=True
)
def calculate_trend_lines(json_data, trend_window_size, trend_slope_threshold):
    if not json_data:
        return None
    
    # 设置默认值，以防参数为None
    if trend_window_size is None:
        trend_window_size = 10
    if trend_slope_threshold is None:
        trend_slope_threshold = 0.001
        
    print(f"计算趋势线 - 窗口大小: {trend_window_size}, 斜率阈值: {trend_slope_threshold}")
    
    try:
        # 解析数据
        data = pd.read_json(json_data, orient='split')
        
        # 处理日期列，兼容不同的命名
        if 'Datetime' in data.columns and 'Date' not in data.columns:
            data = data.rename(columns={'Datetime': 'Date'})
        
        # 确保Date列是日期时间类型并设置为索引
        if 'Date' not in data.columns:
            print("数据中缺少Date列，无法计算趋势线")
            return None
        
        data['Date'] = pd.to_datetime(data['Date'])
        data.set_index('Date', inplace=True)
        
        # 计算趋势线
        try:
            support_lines, resistance_lines = identify_trend_lines(data, window_size=trend_window_size, slope_threshold=trend_slope_threshold)
            
            # 将datetime对象转换为字符串，以便JSON序列化
            support_lines_json = []
            for x1, y1, x2, y2 in support_lines:
                support_lines_json.append({
                    'x1': x1.isoformat(),
                    'y1': float(y1),
                    'x2': x2.isoformat(),
                    'y2': float(y2)
                })
                
            resistance_lines_json = []
            for x1, y1, x2, y2 in resistance_lines:
                resistance_lines_json.append({
                    'x1': x1.isoformat(),
                    'y1': float(y1),
                    'x2': x2.isoformat(),
                    'y2': float(y2)
                })
            
            trend_data = {
                'support_lines': support_lines_json,
                'resistance_lines': resistance_lines_json,
                'last_index': data.index[-1].isoformat() if len(data.index) > 0 else None
            }
            
            return trend_data
            
        except Exception as e:
            print(f"计算趋势线时出错: {str(e)}")
            return None
            
    except Exception as e:
        print(f"趋势线数据计算失败: {str(e)}")
        return None

# 启动服务器
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='启动股票交易信号可视化Web应用')
    parser.add_argument('--port', type=int, default=8050, help='服务器端口号')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='服务器主机地址')
    args = parser.parse_args()
    
    # 使用命令行参数作为配置
    port = int(os.environ.get('PORT', args.port))
    host = os.environ.get('HOST', args.host)
    
    print(f"应用将在 http://{host}:{port}/ 启动")
    app.run(debug=True, port=port, host=host)
else:
    # 用于Vercel部署
    # 注意这是Vercel需要的入口点，不要修改这一行
    # app.server是Flask应用实例
    application = app.server 