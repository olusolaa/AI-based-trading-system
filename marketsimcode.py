import numpy as np
import pandas as pd

from util import get_data


def get_daily_returns(df):
    daily_returns = df.pct_change().fillna(0)
    return daily_returns


def get_portfolio_stats(port_val, daily_rf=0, samples_per_year=252):
    daily_ret = (port_val / port_val.shift(1)) - 1
    daily_ret = daily_ret[1:]
    cr = (port_val.iloc[-1] / port_val.iloc[0]) - 1
    adr = daily_ret.mean()
    sddr = daily_ret.std()
    sr = np.sqrt(samples_per_year) * (adr - daily_rf) / sddr
    return cr, adr, sddr, sr


def compute_portvals(trades_df, start_val=1000000, commission=0.0, impact=0.0):
    symbols = trades_df.columns.tolist()
    start_date = trades_df.index.min()
    end_date = trades_df.index.max()
    prices_all = get_data(symbols, pd.date_range(start_date, end_date))
    prices_all['Cash'] = 1.0

    trades = trades_df.copy()
    trades['Cash'] = 0.0
    trades.iloc[0, -1] = start_val

    for date, trades_on_date in trades.iterrows():
        for symbol in symbols:
            shares = trades_on_date[symbol]
            if shares != 0:
                price = prices_all.at[date, symbol]
                if shares > 0:  # BUY
                    cost = price * shares * (1 + impact) + commission
                    trades.at[date, 'Cash'] -= cost
                elif shares < 0:  # SELL
                    revenue = price * -shares * (1 - impact) - commission
                    trades.at[date, 'Cash'] += revenue

    holdings = trades.cumsum()
    portvals = (holdings * prices_all).sum(axis=1)
    portvals_df = pd.DataFrame(portvals, columns=['Portfolio Value'])

    return portvals_df