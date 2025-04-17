# 股票交易信号可视化应用

这是一个基于Dash和Plotly构建的交互式股票交易信号可视化Web应用，可以分析股票数据，计算技术指标，并显示买入/卖出信号和趋势线。

## 功能特点

- 加载股票历史数据（包括15分钟、30分钟、1小时等多种K线间隔）
- 展示价格、成交量、移动平均线和其它技术指标
- 自动计算并显示支撑线和阻力线（趋势线）
- KDJ、MACD、RSI等技术指标可视化
- 买入/卖出信号显示和回测
- 时间特征分析（小时、星期效应）
- 响应式设计，适配各种设备

## 本地运行

1. 克隆仓库到本地
   ```bash
   git clone https://github.com/yourusername/stock_trading_signal_ml.git
   cd stock_trading_signal_ml
   ```

2. 安装依赖
   ```bash
   pip install -r requirements.txt
   ```

3. 运行应用
   ```bash
   python stock_web_app.py
   ```

4. 在浏览器中访问: http://127.0.0.1:8050/

## 部署到Vercel

### 前提条件

- 注册一个[Vercel账号](https://vercel.com/signup)
- 安装[Vercel CLI](https://vercel.com/download)
- 将代码推送到GitHub仓库

### 部署步骤

1. 登录Vercel CLI
   ```bash
   vercel login
   ```

2. 从项目根目录部署
   ```bash
   vercel
   ```

3. 回答提示问题：
   - 确认部署目录
   - 链接到已存在的项目或创建新项目
   - 确认项目设置

4. 按照终端指示访问部署好的应用URL

### 通过GitHub自动部署

1. 在Vercel控制台中导入项目
2. 选择GitHub仓库
3. 配置项目设置（框架预设选Python）
4. 部署

## 环境变量

在Vercel中可以设置以下环境变量：

- `PORT`: 应用端口号（通常由Vercel自动设置）
- `HOST`: 主机地址（通常由Vercel自动设置）
- `DEBUG`: 设置为"True"启用调试模式

## 许可证

MIT 