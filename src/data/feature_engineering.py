# Technical indicators for financial time series: RSI, MACD, Bollinger Bands, ATR, Volume Ratio.
import pandas as pd
import numpy as np

def calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    return pd.DataFrame({
        'macd': macd,
        'macd_signal': macd_signal,
        'macd_histogram': macd_hist
    })

def calculate_bollinger_bands(prices: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    sma = prices.rolling(window=window).mean()
    std = prices.rolling(window=window).std()
    upper = sma + (std * num_std)
    lower = sma - (std * num_std)
    width = (upper - lower) / sma
    return pd.DataFrame({
        'bb_upper': upper,
        'bb_middle': sma,
        'bb_lower': lower,
        'bb_width': width
    })

def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    atr = true_range.rolling(window=window).mean()
    return atr

def calculate_volume_ratio(volume: pd.Series, window: int = 20) -> pd.Series:
    volume_sma = volume.rolling(window=window).mean()
    volume_ratio = volume / volume_sma
    return volume_ratio

def add_technical_indicators(df: pd.DataFrame, price_col: str, high_col: str = None,
                            low_col: str = None, volume_col: str = None,
                            rsi_window: int = 14, macd_fast: int = 12,
                            macd_slow: int = 26, macd_signal: int = 9,
                            bb_window: int = 20, bb_std: float = 2.0,
                            atr_window: int = 14, volume_window: int = 20) -> pd.DataFrame:
    df = df.copy()
    
    if price_col in df.columns:
        df['rsi'] = calculate_rsi(df[price_col], window=rsi_window)
        macd_data = calculate_macd(df[price_col], fast=macd_fast, slow=macd_slow, signal=macd_signal)
        df = pd.concat([df, macd_data], axis=1)
        bb_data = calculate_bollinger_bands(df[price_col], window=bb_window, num_std=bb_std)
        df = pd.concat([df, bb_data], axis=1)
    
    if all(col in df.columns for col in [high_col, low_col, price_col]) and high_col is not None and low_col is not None:
        df['atr'] = calculate_atr(df[high_col], df[low_col], df[price_col], window=atr_window)
    
    if volume_col is not None and volume_col in df.columns:
        df['volume_ratio'] = calculate_volume_ratio(df[volume_col], window=volume_window)
    
    return df
