import numpy as np
import pandas as pd
import datetime as dt
import util as ut
from QLearner import QLearner
import indicators as ind

class StrategyLearner(object):

    def __init__(self, verbose=False, impact=0.005, commission=9.95, alpha=0.01, gamma=0.7, rar=0.5, radr=0.99):
        self.verbose = verbose
        self.impact = impact
        self.commission = commission
        self.learner = QLearner(num_states=625, num_actions=3, alpha=alpha, gamma=gamma, rar=rar, radr=radr)

    def add_evidence(self, symbol="AAPL", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000):
        dates = pd.date_range(sd, ed)
        prices = ut.get_data([symbol], dates)
        prices = prices[symbol]

        indicators = self.get_indicators(prices)
        states = self.compute_states(indicators)

        num_days = prices.shape[0]
        port_val = sv

        for iteration in range(50):  # Extended to ensure convergence
            state = states[0]
            self.learner.querysetstate(state)
            holdings = 0

            for i in range(1, num_days):
                action = self.learner.query(state, reward if i > 1 else 0)

                reward = 0
                trade = 0
                if action == 0:  # Long
                    if holdings == 0:
                        trade = 1000
                        reward = (prices.iloc[i] - prices.iloc[i - 1]) * 1000 - (self.impact * prices.iloc[i] * 1000) - self.commission
                    elif holdings == -1000:
                        trade = 2000
                        reward = 2 * (prices.iloc[i] - prices.iloc[i - 1]) * 1000 - (self.impact * prices.iloc[i] * 2000) - self.commission
                elif action == 1:  # Short
                    if holdings == 0:
                        trade = -1000
                        reward = (prices.iloc[i - 1] - prices.iloc[i]) * 1000 - (self.impact * prices.iloc[i] * 1000) - self.commission
                    elif holdings == 1000:
                        trade = -2000
                        reward = 2 * (prices.iloc[i - 1] - prices.iloc[i]) * 1000 - (self.impact * prices.iloc[i] * 2000) - self.commission
                elif action == 2:  # Cash
                    if holdings == 1000:
                        trade = -1000
                        reward = (prices.iloc[i - 1] - prices.iloc[i]) * 1000 - (self.impact * prices.iloc[i] * 1000) - self.commission
                    elif holdings == -1000:
                        trade = 1000
                        reward = (prices.iloc[i] - prices.iloc[i - 1]) * 1000 - (self.impact * prices.iloc[i] * 1000) - self.commission

                holdings += trade
                port_val += reward
                state = states[i]

    def testPolicy(self, symbol="AAPL", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31), sv=100000):
        dates = pd.date_range(sd, ed)
        prices = ut.get_data([symbol], dates)
        prices = prices[symbol]

        indicators = self.get_indicators(prices)
        states = self.compute_states(indicators)

        num_days = prices.shape[0]
        trades = pd.DataFrame(index=prices.index, columns=[symbol])
        trades[symbol] = 0

        state = states[0]
        self.learner.querysetstate(state)
        holdings = 0
        reward = 0

        for i in range(1, num_days):
            action = self.learner.querysetstate(state) if i == 0 else self.learner.query(state, reward)

            trade = 0
            reward = 0
            if action == 0:  # Long
                if holdings == 0:
                    trade = 1000
                elif holdings == -1000:
                    trade = 2000
            elif action == 1:  # Short
                if holdings == 0:
                    trade = -1000
                elif holdings == 1000:
                    trade = -2000
            elif action == 2:  # Cash
                trade = -holdings

            holdings += trade
            trades.iloc[i] = trade
            state = states[i]

        return trades

    def get_indicators(self, prices):
        sma = ind.simple_ma(prices, window=20)
        bbp = ind.bollinger_bands(prices, window=20, threshold=2)
        macd_hist = ind.macd(prices, short_window=12, long_window=26, signal_window=9)
        stochastic = ind.stochastic_oscillator(prices, k_window=14, d_window=3)
        indicators = pd.concat([sma, bbp, macd_hist, stochastic], axis=1)
        indicators.columns = ["SMA", "BBP", "MACD_Hist", "Stochastic"]
        return indicators

    def compute_states(self, indicators):
        bins = [5, 5, 5, 5]  # Updated bin sizes to 5
        digitized = np.digitize(indicators.values, bins)
        states = np.sum(digitized * [125, 25, 5, 1], axis=1)
        return states

    def author(self):
        return 'oalao30'
