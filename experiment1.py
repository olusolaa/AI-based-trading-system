import ManualStrategy as ms
import StrategyLearner as sl
from marketsimcode import compute_portvals, get_data
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import os


def experiment1():
    symbol = "JPM"
    sv = 100000

    # Define the date ranges
    sd_insample = dt.datetime(2008, 1, 1)
    ed_insample = dt.datetime(2009, 12, 31)
    sd_outsample = dt.datetime(2010, 1, 1)
    ed_outsample = dt.datetime(2011, 12, 31)

    # Create output directory if it doesn't exist
    output_dir = "images"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Manual Strategy In-Sample
    manual = ms.ManualStrategy()
    manual_trades_insample = manual.testPolicy(symbol=symbol, sd=sd_insample, ed=ed_insample, sv=sv)
    manual_portvals_insample = compute_portvals(manual_trades_insample, start_val=sv)

    # Manual Strategy Out-of-Sample
    manual_trades_outsample = manual.testPolicy(symbol=symbol, sd=sd_outsample, ed=ed_outsample, sv=sv)
    manual_portvals_outsample = compute_portvals(manual_trades_outsample, start_val=sv)

    # Strategy Learner In-Sample
    learner = sl.StrategyLearner()
    learner.add_evidence(symbol=symbol, sd=sd_insample, ed=ed_insample, sv=sv)
    learner_trades_insample = learner.testPolicy(symbol=symbol, sd=sd_insample, ed=ed_insample, sv=sv)
    learner_portvals_insample = compute_portvals(learner_trades_insample, start_val=sv)

    # Strategy Learner Out-of-Sample
    learner_trades_outsample = learner.testPolicy(symbol=symbol, sd=sd_outsample, ed=ed_outsample, sv=sv)
    learner_portvals_outsample = compute_portvals(learner_trades_outsample, start_val=sv)

    # Benchmark In-Sample
    prices_insample = get_data([symbol], pd.date_range(sd_insample, ed_insample))
    prices_insample = prices_insample[symbol]
    benchmark_portvals_insample = (prices_insample / prices_insample.iloc[0]) * sv

    # Benchmark Out-of-Sample
    prices_outsample = get_data([symbol], pd.date_range(sd_outsample, ed_outsample))
    prices_outsample = prices_outsample[symbol]
    benchmark_portvals_outsample = (prices_outsample / prices_outsample.iloc[0]) * sv

    # Plot results for In-Sample
    plt.figure(figsize=(10, 6))
    plt.plot(benchmark_portvals_insample, label="Benchmark", color='purple')
    plt.plot(manual_portvals_insample, label="Manual Strategy", color='red')
    plt.plot(learner_portvals_insample, label="Strategy Learner", color='blue')
    plt.title("Experiment 1: In-Sample Comparison")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, "in_sample_comparison.png"))

    # Plot results for Out-of-Sample
    plt.figure(figsize=(10, 6))
    plt.plot(benchmark_portvals_outsample, label="Benchmark", color='purple')
    plt.plot(manual_portvals_outsample, label="Manual Strategy", color='red')
    plt.plot(learner_portvals_outsample, label="Strategy Learner", color='blue')
    plt.title("Experiment 1: Out-of-Sample Comparison")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, "out_of_sample_comparison.png"))


if __name__ == "__main__":
    experiment1()
