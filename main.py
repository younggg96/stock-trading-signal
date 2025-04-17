from data_fetch.polygon_api import fetch_polygon_15min_data
from features.technical_indicators import compute_technical_indicators
from model.train_model import train_model
from model.signal_generator import generate_signals
import os
import sys

def main():
    # 尝试从环境变量获取 API 密钥
    API_KEY = os.environ.get("POLYGON_API_KEY")
    
    # 如果环境变量中没有，提示用户输入
    if not API_KEY:
        print("Polygon.io API key not found in environment variables.")
        print("You can set it by running: export POLYGON_API_KEY='your_api_key'")
        print("Or enter your API key now:")
        API_KEY = input().strip()
        
        if not API_KEY:
            print("Error: API key is required to fetch data from Polygon.io")
            print("Get your free API key at: https://polygon.io/")
            sys.exit(1)
    
    SYMBOL = "NVDA"
    print(f"Using API key: {API_KEY[:5]}...{API_KEY[-4:]}")
    print(f"Fetching data for {SYMBOL}...")
    
    try:
        df = fetch_polygon_15min_data(SYMBOL, API_KEY)
        
        if df.empty:
            print("Error: No data returned from API. Please check your API key and try again.")
            sys.exit(1)
            
        print("Calculating indicators...")
        df = compute_technical_indicators(df)

        print("Training model...")
        train_model(df)

        print("Generating signals...")
        df_with_signals = generate_signals(df)
        print(df_with_signals.tail(10))
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Please check your API key and internet connection.")
        sys.exit(1)

if __name__ == "__main__":
    main()