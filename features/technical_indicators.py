import pandas as pd

def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df['ma_10'] = df['close'].rolling(window=10).mean()
    df['rsi'] = compute_rsi(df['close'], window=14)
    df['macd'], df['macd_signal'] = compute_macd(df['close'])
    df.dropna(inplace=True)
    return df

def compute_rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_macd(series):
    exp1 = series.ewm(span=12, adjust=False).mean()
    exp2 = series.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal