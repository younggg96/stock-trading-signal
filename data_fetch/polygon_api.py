import requests
import pandas as pd
import json
from datetime import datetime, timedelta

def fetch_polygon_15min_data(symbol: str, api_key: str, limit=1000):
    # 使用当前日期，并只获取最近2天的数据（适应免费计划）
    end_date = datetime.now()
    start_date = end_date - timedelta(days=2)  # 免费 API 一般只提供最近的数据
    
    # 格式化日期为 YYYY-MM-DD
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    
    print(f"Requesting data from {start_date_str} to {end_date_str}")
    
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/15/minute/{start_date_str}/{end_date_str}"
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": limit,
        "apiKey": api_key
    }
    response = requests.get(url, params=params)
    
    # Print response for debugging
    print(f"API Status Code: {response.status_code}")
    
    data = response.json()
    
    # Print the structure of the response
    print("API Response Keys:", list(data.keys()))
    
    # Check if results exist
    if "results" not in data:
        print("Error in API response:", data.get("error", data.get("message", "Unknown error")))
        print("Full response:", json.dumps(data, indent=2))
        raise ValueError(f"Failed to get data from Polygon API: {data.get('error', data.get('message', 'Unknown error'))}")
    
    results = data.get("results", [])
    
    if not results:
        print("No results returned from API")
        return pd.DataFrame()
    
    # Print first result structure
    if results:
        print("First result structure:", list(results[0].keys()))
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Check if 't' column exists
    if 't' in df.columns:
        df['t'] = pd.to_datetime(df['t'], unit='ms')
        df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume", "t": "timestamp"})
    else:
        print("Column 't' not found in API response. Available columns:", df.columns.tolist())
        # Try to identify timestamp column
        timestamp_candidates = [col for col in df.columns if any(time_str in str(col).lower() for time_str in ['time', 'date', 'timestamp'])]
        if timestamp_candidates:
            print(f"Possible timestamp column found: {timestamp_candidates[0]}")
            df['timestamp'] = pd.to_datetime(df[timestamp_candidates[0]], unit='ms')
        
        # Rename other columns if they exist
        col_mapping = {"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
        for old_col, new_col in col_mapping.items():
            if old_col in df.columns:
                df = df.rename(columns={old_col: new_col})
    
    return df