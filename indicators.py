import pandas as pd

def author():
    return "oalao30"

def simple_ma(df, window=5):
    mean = df.rolling(window=window).mean()
    if isinstance(mean, pd.Series):
        mean = mean.to_frame(name='mean')
    return mean

def bollinger_bands(df, window=20, threshold=2):
    mean = df.rolling(window=window).mean()
    std = df.rolling(window=window).std()
    upper = mean + threshold * std
    lower = mean - threshold * std
    bbp = (df - mean) / (upper - lower)
    if isinstance(bbp, pd.Series):
        bbp = bbp.to_frame(name='bbp')
    return bbp

def macd(price, short_window=12, long_window=26, signal_window=9):
    ema_short = price.ewm(span=short_window, adjust=False).mean()
    ema_long = price.ewm(span=long_window, adjust=False).mean()
    macd_line = ema_short - ema_long
    signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()
    macd_hist = macd_line - signal_line
    if isinstance(macd_hist, pd.Series):
        macd_hist = macd_hist.to_frame(name='macd_hist')
    return macd_hist

def stochastic_oscillator(df, k_window=14, d_window=3):
    high = df.rolling(window=k_window).max()
    low = df.rolling(window=k_window).min()
    K = 100 * (df - low) / (high - low)
    if isinstance(K, pd.Series):
        K = K.to_frame(name='K')
    return K
