import ManualStrategy as ms
import StrategyLearner as sl
from marketsimcode import compute_portvals, get_data
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import os

def experiment2():
    symbol = "JPM"
    sv = 100000

    # Define the date ranges
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)

    # # Define different impact values
    # impact_values = [0.0, 0.005, 0.01, 0.02]
    #
    # # Create output directory if it doesn't exist
    # output_dir = "images"
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    #
    # # Store results
    # results = {}
    #
    # for impact in impact_values:
    #     learner = sl.StrategyLearner(impact=impact)
    #     learner.add_evidence(symbol=symbol, sd=sd, ed=ed, sv=sv)
    #     trades = learner.testPolicy(symbol=symbol, sd=sd, ed=ed, sv=sv)
    #     portvals = compute_portvals(trades, start_val=sv)
    #     if portvals is not None and not portvals.empty:
    #         results[impact] = portvals
    #
    # # Benchmark
    # prices = get_data([symbol], pd.date_range(sd, ed))
    # prices = prices[symbol]
    # benchmark_portvals = (prices / prices.iloc[0]) * sv
    #
    # # Plot portfolio values over time
    # plt.figure(figsize=(10, 6))
    # for impact in impact_values:
    #     if impact in results:
    #         plt.plot(results[impact], label=f"Impact: {impact}")
    # plt.plot(benchmark_portvals, label="Benchmark", color='purple')
    # plt.title("Experiment 2: Impact Analysis - Portfolio Value Over Time")
    # plt.xlabel("Date")
    # plt.ylabel("Portfolio Value")
    # plt.legend()
    # plt.grid()
    # plt.savefig(os.path.join(output_dir, "impact_analysis_portfolio_values.png"))
    #
    # # Calculate metrics
    # metrics_list = []
    #
    # for impact in impact_values:
    #     if impact in results:
    #         portvals = results[impact]
    #         daily_returns = portvals.pct_change().dropna()
    #         cumulative_return = (portvals.iloc[-1] / portvals.iloc[0]) - 1
    #         stdev_daily_returns = daily_returns.std()
    #         mean_daily_returns = daily_returns.mean()
    #
    #         metrics_list.append({
    #             "Impact": impact,
    #             "Cumulative Return": cumulative_return,
    #             "Stdev of Daily Returns": stdev_daily_returns,
    #             "Mean of Daily Returns": mean_daily_returns
    #         })
    #
    # metrics = pd.DataFrame(metrics_list)
    #
    # # Convert metrics to numeric types
    # metrics["Cumulative Return"] = pd.to_numeric(metrics["Cumulative Return"])
    # metrics["Stdev of Daily Returns"] = pd.to_numeric(metrics["Stdev of Daily Returns"])
    # metrics["Mean of Daily Returns"] = pd.to_numeric(metrics["Mean of Daily Returns"])
    #
    # # Check the DataFrame
    # print(metrics)
    #
    # # Plot metrics
    # metrics.set_index("Impact", inplace=True)
    #
    # # Cumulative Return
    # plt.figure(figsize=(10, 6))
    # metrics["Cumulative Return"].plot(kind='bar')
    # plt.title("Experiment 2: Impact Analysis - Cumulative Return")
    # plt.xlabel("Impact")
    # plt.ylabel("Cumulative Return")
    # plt.grid()
    # plt.savefig(os.path.join(output_dir, "impact_analysis_cumulative_return.png"))
    #
    # # Standard Deviation of Daily Returns
    # plt.figure(figsize=(10, 6))
    # metrics["Stdev of Daily Returns"].plot(kind='bar')
    # plt.title("Experiment 2: Impact Analysis - Stdev of Daily Returns")
    # plt.xlabel("Impact")
    # plt.ylabel("Standard Deviation of Daily Returns")
    # plt.grid()
    # plt.savefig(os.path.join(output_dir, "impact_analysis_stdev_daily_returns.png"))
    #
    # # Mean of Daily Returns
    # plt.figure(figsize=(10, 6))
    # metrics["Mean of Daily Returns"].plot(kind='bar')
    # plt.title("Experiment 2: Impact Analysis - Mean of Daily Returns")
    # plt.xlabel("Impact")
    # plt.ylabel("Mean of Daily Returns")
    # plt.grid()
    # plt.savefig(os.path.join(output_dir, "impact_analysis_mean_daily_returns.png"))

if __name__ == "__main__":
    experiment2()
