import numpy as np
import pandas as pd
import datetime as dt
import util as ut
from QLearner import QLearner
import indicators as ind
from StrategyLearner import StrategyLearner
from marketsimcode import compute_portvals
from concurrent.futures import ProcessPoolExecutor, as_completed


def evaluate_strategy(learner_params, in_sample_period, out_sample_period, symbol="AAPL", sv=100000):
    learner = QLearner(
        num_states=10000, num_actions=3,
        alpha=learner_params['alpha'],
        gamma=learner_params['gamma'],
        rar=learner_params['rar'],
        radr=learner_params['radr']
    )
    strategy = StrategyLearner(verbose=False, impact=0.005, commission=9.95)
    strategy.learner = learner

    # Train the learner
    strategy.add_evidence(symbol=symbol, sd=in_sample_period[0], ed=in_sample_period[1], sv=sv)

    # Test the learner
    trades = strategy.testPolicy(symbol=symbol, sd=out_sample_period[0], ed=out_sample_period[1], sv=sv)
    portvals = compute_portvals(trades, start_val=sv, commission=9.95, impact=0.005)

    # Compute performance metrics
    cumulative_return = (portvals.iloc[-1] / portvals.iloc[0]) - 1
    daily_returns = portvals.pct_change().dropna()
    sharpe_ratio = (np.sqrt(252) * daily_returns.mean() / daily_returns.std()).iloc[0]
    sortino_ratio = (np.sqrt(252) * daily_returns.mean() / daily_returns[daily_returns < 0].std()).iloc[0]
    max_drawdown = ((portvals / portvals.cummax()) - 1).min().iloc[0]
    volatility = daily_returns.std().iloc[0]

    return cumulative_return, sharpe_ratio, sortino_ratio, max_drawdown, volatility, learner_params


def main():
    # Define the parameter grid
    param_grid = {
        'alpha': [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
        'gamma': [0.7, 0.8, 0.85, 0.9, 0.95, 0.99],
        'rar': [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
        'radr': [0.9, 0.93, 0.95, 0.97, 0.99]
    }

    # Define the in-sample and out-sample periods
    in_sample_period = (dt.datetime(2008, 1, 1), dt.datetime(2009, 12, 31))
    out_sample_period = (dt.datetime(2010, 1, 1), dt.datetime(2011, 12, 31))

    # Perform grid search
    best_params = None
    best_performance = (-np.inf, -np.inf, -np.inf, np.inf,
                        np.inf)  # (cumulative_return, sharpe_ratio, sortino_ratio, max_drawdown, volatility)

    # Prepare the parameter combinations
    param_combinations = [
        {'alpha': alpha, 'gamma': gamma, 'rar': rar, 'radr': radr}
        for alpha in param_grid['alpha']
        for gamma in param_grid['gamma']
        for rar in param_grid['rar']
        for radr in param_grid['radr']
    ]

    # Use ProcessPoolExecutor for parallel execution
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(evaluate_strategy, params, in_sample_period, out_sample_period) for params in
                   param_combinations]

        for future in as_completed(futures):
            cumulative_return, sharpe_ratio, sortino_ratio, max_drawdown, volatility, learner_params = future.result()

            # Update best parameters based on multiple metrics
            if sharpe_ratio > best_performance[1] and sortino_ratio > best_performance[2] and max_drawdown < \
                    best_performance[3] and volatility < best_performance[4]:
                best_params = learner_params
                best_performance = (cumulative_return, sharpe_ratio, sortino_ratio, max_drawdown, volatility)

    print("Best Parameters:", best_params)
    print("Best Performance:", best_performance)


if __name__ == "__main__":
    main()
