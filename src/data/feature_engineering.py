"""
Technical indicators and feature engineering for financial time series.
"""
import pandas as pd
import numpy as np


def calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).
    
    Parameters:
    -----------
    prices : pd.Series
        Price series
    window : int
        Lookback window (default 14)
    
    Returns:
    --------
    pd.Series
        RSI values (0-100)
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """
    Calculate MACD (Moving Average Convergence Divergence).
    
    Parameters:
    -----------
    prices : pd.Series
        Price series
    fast : int
        Fast EMA period (default 12)
    slow : int
        Slow EMA period (default 26)
    signal : int
        Signal line EMA period (default 9)
    
    Returns:
    --------
    pd.DataFrame
        Columns: 'macd', 'macd_signal', 'macd_histogram'
    """
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
    """
    Calculate Bollinger Bands.
    
    Parameters:
    -----------
    prices : pd.Series
        Price series
    window : int
        Moving average window (default 20)
    num_std : float
        Number of standard deviations (default 2.0)
    
    Returns:
    --------
    pd.DataFrame
        Columns: 'bb_upper', 'bb_middle', 'bb_lower', 'bb_width'
    """
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
    """
    Calculate Average True Range (ATR).
    
    Parameters:
    -----------
    high : pd.Series
        High prices
    low : pd.Series
        Low prices
    close : pd.Series
        Close prices
    window : int
        Smoothing window (default 14)
    
    Returns:
    --------
    pd.Series
        ATR values
    """
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    atr = true_range.rolling(window=window).mean()
    
    return atr


def calculate_volume_ratio(volume: pd.Series, window: int = 20) -> pd.Series:
    """
    Calculate volume to moving average ratio.
    
    Parameters:
    -----------
    volume : pd.Series
        Volume series
    window : int
        Moving average window (default 20)
    
    Returns:
    --------
    pd.Series
        Volume ratio (volume / SMA)
    """
    volume_sma = volume.rolling(window=window).mean()
    volume_ratio = volume / volume_sma
    return volume_ratio


def add_technical_indicators(df: pd.DataFrame, price_col: str, high_col: str = None,
                            low_col: str = None, volume_col: str = None,
                            rsi_window: int = 14, macd_fast: int = 12,
                            macd_slow: int = 26, macd_signal: int = 9,
                            bb_window: int = 20, bb_std: float = 2.0,
                            atr_window: int = 14, volume_window: int = 20) -> pd.DataFrame:
    """
    Add technical indicators to a DataFrame.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame with price data
    price_col : str
        Column name for price/close
    high_col : str, optional
        Column name for high prices (required for ATR)
    low_col : str, optional
        Column name for low prices (required for ATR)
    volume_col : str, optional
        Column name for volume (required for volume ratio)
    rsi_window : int
        RSI window period
    macd_fast, macd_slow, macd_signal : int
        MACD parameters
    bb_window : int
        Bollinger Bands window
    bb_std : float
        Bollinger Bands standard deviations
    atr_window : int
        ATR window period
    volume_window : int
        Volume SMA window
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with technical indicators added
    """
    df = df.copy()
    
    # RSI
    if price_col in df.columns:
        df['rsi'] = calculate_rsi(df[price_col], window=rsi_window)
    
    # MACD
    if price_col in df.columns:
        macd_data = calculate_macd(df[price_col], fast=macd_fast, slow=macd_slow, signal=macd_signal)
        df = pd.concat([df, macd_data], axis=1)
    
    # Bollinger Bands
    if price_col in df.columns:
        bb_data = calculate_bollinger_bands(df[price_col], window=bb_window, num_std=bb_std)
        df = pd.concat([df, bb_data], axis=1)
    
    # ATR (requires high, low, close)
    if all(col in df.columns for col in [high_col, low_col, price_col]) and \
       high_col is not None and low_col is not None:
        df['atr'] = calculate_atr(df[high_col], df[low_col], df[price_col], window=atr_window)
    
    # Volume Ratio
    if volume_col is not None and volume_col in df.columns:
        df['volume_ratio'] = calculate_volume_ratio(df[volume_col], window=volume_window)
    
    return df

