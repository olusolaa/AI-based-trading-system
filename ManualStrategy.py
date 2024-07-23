import pandas as pd
import numpy as np
import datetime as dt
import indicators as ind
from util import get_data

class ManualStrategy(object):

    def __init__(self, verbose=False):
        self.verbose = verbose

    def testPolicy(self, symbol="AAPL", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31), sv=100000):
        dates = pd.date_range(sd, ed)
        prices = get_data([symbol], dates)
        prices = prices[symbol]

        sma = ind.simple_ma(prices)
        bbp = ind.bollinger_bands(prices)
        macd = ind.macd(prices)
        stochastic = ind.stochastic_oscillator(prices)

        trades = pd.DataFrame(index=prices.index, columns=[symbol])
        trades[symbol] = 0

        holdings = 0  # Start with no holdings

        for i in range(1, len(prices)):
            signal = 0
            if (sma.iloc[i]['mean'] > prices.iloc[i] and
                bbp.iloc[i]['bbp'] > 1 and
                macd.iloc[i]['macd_hist'] > 0 and
                stochastic.iloc[i]['K'] > 80):
                signal = -1  # Overbought
            elif (sma.iloc[i]['mean'] < prices.iloc[i] and
                  bbp.iloc[i]['bbp'] < 0 and
                  macd.iloc[i]['macd_hist'] < 0 and
                  stochastic.iloc[i]['K'] < 20):
                signal = 1  # Oversold

            trade = 0
            if signal == 1:
                if holdings == 0:
                    trade = 1000
                elif holdings == -1000:
                    trade = 2000
            elif signal == -1:
                if holdings == 0:
                    trade = -1000
                elif holdings == 1000:
                    trade = -2000

            holdings += trade
            trades.iloc[i] = trade

        return trades

    def author(self):
        return 'oalao30'  # Replace with your GT username
