import joblib
import pandas as pd

def generate_signals(df: pd.DataFrame):
    model = joblib.load("model/xgb_model.pkl")
    features = ['ma_10', 'rsi', 'macd', 'macd_signal']
    df['pred'] = model.predict(df[features])
    df['signal'] = df['pred'].map({1: 'BUY', 0: 'HOLD'})
    return df[['timestamp', 'close', 'signal']]