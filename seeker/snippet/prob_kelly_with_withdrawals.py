#date: 2024-12-06T17:10:31Z
#url: https://api.github.com/gists/f8d8692cba1455f046ba9784b0fcab0b
#owner: https://api.github.com/users/robcarver17

from copy import copy

import matplotlib
import numpy as np

matplotlib.use("TkAgg")
matplotlib.rcParams.update({"font.size": 22})

from numpy.random import normal
import pandas as pd
from syscore.interactive.progress_bar import progressBar ## dont' have to use this if you dont' have pysystemtrade



ann_risk_free = 0.05
daily_rf = ann_risk_free / 256

years_to_run_over = 10
days_to_run_over = years_to_run_over * 256

def series_of_rand_returns():
    return pd.Series([rand_single_return() for __ in range(days_to_run_over)])

def rand_single_return():
    return normal(daily_mean, daily_stdev)

def leverage_series_of_returns(some_returns: pd.Series, leverage: float):
    return leverage * some_returns - (daily_rf*(leverage-1))

def adjust_returns_so_hits_expectation_exactly(some_returns: pd.Series):
    emp_std = some_returns.std()
    adjustment = daily_stdev / emp_std
    some_returns = some_returns*adjustment
    emp_mean = some_returns.mean()
    adjustment = daily_mean - emp_mean

    return some_returns+adjustment

def final_value(some_returns: pd.Series):
    x = 1+some_returns
    cum_x = x.cumprod()

    return cum_x.values[-1]

def geo_mean(some_returns: pd.Series):
    final_x = final_value(some_returns)

    daily_geo_return = final_x**(1/len(some_returns)) -1

    ann_geo_return = 256*daily_geo_return

    return ann_geo_return

def final_value_with_withdrawals(some_returns: pd.Series, withdrawal_annual_rate: float):
    daily_withdrawal_rate = withdrawal_annual_rate/256
    x = 1+some_returns-daily_withdrawal_rate
    cum_x = x.cumprod()

    return cum_x.values[-1]


def find_optimal_leverage(some_returns: pd.Series) -> float:
    leveraged_returns = dict([
        (leverage, leverage_series_of_returns(some_returns=some_returns, leverage=leverage))
        for leverage in all_leverage_values_for_plots
    ])
    final_values_at_zero_with_rate = [final_value(series)
                                      for series in list(leveraged_returns.values())]
    max_final = np.max(final_values_at_zero_with_rate)
    index_max = final_values_at_zero_with_rate.index(max_final)
    optimal_leverage = all_leverage_values_for_plots[index_max]

    return optimal_leverage


def find_max_safe_rate_to_withdraw(some_returns: pd.Series, optimal_leverage: float, min_value_of_final_value: float):
    max_withdrawal_rate = 0
    current_withdrawalrate = 0
    okay = True

    while okay:
        if current_withdrawalrate<0.1:
            current_withdrawalrate+=0.001
        else:
            current_withdrawalrate+=0.01

        final_values_at_optimal_leverage = \
        final_value_with_withdrawals(leverage_series_of_returns(some_returns=some_returns, leverage=optimal_leverage),
                                     withdrawal_annual_rate=current_withdrawalrate)

        if final_values_at_optimal_leverage<min_value_of_final_value:
            break

        max_withdrawal_rate=copy(current_withdrawalrate)

    return max_withdrawal_rate




arith_ann_mean = 0.15
ann_std_dev = 0.20
daily_stdev = ann_std_dev / 16
daily_mean = arith_ann_mean / 256

some_returns = adjust_returns_so_hits_expectation_exactly(series_of_rand_returns())

all_leverage_values_for_plots = list(np.arange(0.1,20, .1))

leveraged_returns = dict([
    (leverage, leverage_series_of_returns(some_returns=some_returns, leverage=leverage))
    for leverage in all_leverage_values_for_plots
])

def raw_leveraged(some_returns, leverage):
    return some_returns*leverage

import numpy as np

raw_means = [raw_leveraged(some_returns, leverage).mean()*256 for leverage in  all_leverage_values_for_plots]
arith_means = [series.mean()*256 for series in list(leveraged_returns.values())]
arith_std = [series.std()*16 for series in list(leveraged_returns.values())]
final_values = [final_value(series) for series in list(leveraged_returns.values())]
geo_means = [geo_mean(series) for series in list(leveraged_returns.values())]

to_plot = pd.DataFrame(dict(raw_mean = raw_means, mean = arith_means,
                            stdev = arith_std, geo_mean = geo_means
                            ), index = all_leverage_values_for_plots)

to_plot.plot()

final_plot = pd.DataFrame(dict(geo_mean=geo_means[:50], final=final_values[:50]), index=all_leverage_values_for_plots[:50])
final_plot['geo_mean'].plot()
final_plot['final'].plot(secondary_y=True)

## show some intution around max geo means and final values for different SR

final_values_SR05 = final_values
geo_means_05 = geo_means
stdev_05 = arith_std

## Another example
arith_ann_mean = 0.15
ann_risk_free = 0.05
ann_std_dev = 0.10

daily_mean = arith_ann_mean / 256
daily_stdev = ann_std_dev / 16

some_returns = adjust_returns_so_hits_expectation_exactly(series_of_rand_returns())
leveraged_returns = dict([
    (leverage, leverage_series_of_returns(some_returns=some_returns, leverage=leverage))

    for leverage in all_leverage_values_for_plots
])


final_values_1 = [final_value(series) for series in list(leveraged_returns.values())]
geo_means_1 = [geo_mean(series) for series in list(leveraged_returns.values())]
arith_std_1 = [series.std()*16 for series in list(leveraged_returns.values())]


both_geo_against_lev = pd.DataFrame(dict(a=geo_means_05, b=geo_means_1), index=all_leverage_values_for_plots)
both_geo_against_std= pd.concat([pd.Series(geo_means_05, index=stdev_05),
                                              pd.Series(geo_means_1, index=arith_std_1),
                                             ], axis=1)


### resampling
#back to original
arith_ann_mean = 0.15
ann_std_dev = 0.20
daily_stdev = ann_std_dev / 16
daily_mean = arith_ann_mean / 256

all_leverage_values_for_plots = list(np.arange(0.1,10, .1)) ## dont need such a big range


monte_count = 1000
all_results = {}
p = progressBar(len(all_leverage_values_for_plots)*monte_count)
for __ in range(monte_count):
    sample_returns = series_of_rand_returns()
    for leverage in all_leverage_values_for_plots:
        leverage = np.round(leverage, 2)  ## werid floating point nonsense
        final_values_at_this_leverage_level= all_results.get(leverage, [])
        p.iterate()
        leveraged_returns = leverage_series_of_returns(some_returns=sample_returns, leverage=leverage)
        final_this_leverage = final_value(leveraged_returns)

        final_values_at_this_leverage_level.append(final_this_leverage)

        all_results[leverage] = final_values_at_this_leverage_level


q_results = {}
all_quantiles = [0.1,0.2,0.3,0.5, 0.75]
for leverage in all_leverage_values_for_plots:
    q_this_leverage = []
    for quantile_point in all_quantiles:
        leverage = np.round(leverage, 2) ## werid floating point nonsense
        point_value = pd.Series(all_results[leverage]).quantile(quantile_point)
        q_this_leverage.append(point_value)

    q_results[leverage] = q_this_leverage

q_to_plot =pd.DataFrame(q_results).transpose()
q_to_plot.columns = all_quantiles


## with withdrawals


arith_ann_mean = 0.15
ann_std_dev = 0.20
daily_stdev = ann_std_dev / 16
daily_mean = arith_ann_mean / 256
withdrawal_rates= [0, 0.01, 0.02, 0.05, 0.1, 0.17, 0.2, 0.5]

some_returns = adjust_returns_so_hits_expectation_exactly(series_of_rand_returns())
leveraged_returns = dict([
    (leverage, leverage_series_of_returns(some_returns=some_returns, leverage=leverage))

    for leverage in all_leverage_values_for_plots
])

all_final_values = {}
for with_rate in withdrawal_rates:
    final_values_at_this_with_rate = [final_value_with_withdrawals(series, withdrawal_annual_rate=with_rate)
                                      for series in list(leveraged_returns.values())]

    all_final_values[with_rate] = final_values_at_this_with_rate


to_plot = pd.DataFrame(all_final_values)
to_plot.columns = withdrawal_rates
to_plot.index = all_leverage_values_for_plots

## find maximum possible withdrawal rate with some condition

ann_std_dev = 0.20
daily_stdev = ann_std_dev / 16
sr_options = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
all_leverage_values_for_plots = list(np.arange(0.1,10, .1)) ## dont need such a big range

years_to_run_over = 10
days_to_run_over = years_to_run_over * 256


min_value_of_final_value = 1.0

all_results = {}
for sr in sr_options:
    arith_ann_mean = (ann_std_dev * sr) + ann_risk_free
    daily_mean = arith_ann_mean / 256

    some_returns = adjust_returns_so_hits_expectation_exactly(series_of_rand_returns())
    leveraged_returns = dict([
        (leverage, leverage_series_of_returns(some_returns=some_returns, leverage=leverage))
    for leverage in all_leverage_values_for_plots
    ])
    final_values_at_zero_with_rate = [final_value(series)
                                      for series in list(leveraged_returns.values())]
    max_final = np.max(final_values_at_zero_with_rate)
    index_max = final_values_at_zero_with_rate.index(max_final)
    optimal_leverage = all_leverage_values_for_plots[index_max]

    max_withdraw = find_max_safe_rate_to_withdraw(some_returns=some_returns, optimal_leverage=optimal_leverage, min_value_of_final_value=min_value_of_final_value)

    all_results[sr] = max_withdraw

#### with different numbers of years and capital


ann_std_dev = 0.20
daily_stdev = ann_std_dev / 16
sr = 1.0
all_leverage_values_for_plots = list(np.arange(0.1,10, .1)) ## dont need such a big range

all_results = {}
for years_to_run_over in [10,20,40, 60]:
    days_to_run_over = years_to_run_over * 256
    results_this_year_count = {}
    for min_value_of_final_value in [0.25, 0.5, 1.0, 2.0, 3.0]:
        arith_ann_mean = (ann_std_dev * sr) + ann_risk_free
        daily_mean = arith_ann_mean / 256

        some_returns = adjust_returns_so_hits_expectation_exactly(series_of_rand_returns())
        optimal_leverage =find_optimal_leverage(some_returns)
        max_withdraw = find_max_safe_rate_to_withdraw(some_returns=some_returns, optimal_leverage=optimal_leverage, min_value_of_final_value=min_value_of_final_value)

        results_this_year_count[min_value_of_final_value] = max_withdraw

    all_results[years_to_run_over] = results_this_year_count

to_plot = pd.DataFrame(all_results)
to_plot.plot()
matplotlib.pyplot.title("Sharpe Ratio %.1f" % sr)


## probabilistic withdrawals

monte_count = 1000
ann_std_dev = 0.20
daily_stdev = ann_std_dev / 16
sr_options = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
p = progressBar(len(sr_options)*monte_count)


all_leverage_values_for_plots = list(np.arange(0.1,10, .1)) ## dont need such a big range

years_to_run_over = 30
days_to_run_over = years_to_run_over * 256
min_value_of_final_value = 1.0

all_results = {}
for sr in sr_options:
    arith_ann_mean = (ann_std_dev * sr) + ann_risk_free
    daily_mean = arith_ann_mean / 256
    results_this_sr = []
    for __ in range(monte_count):
        p.iterate()
        some_returns = series_of_rand_returns()
        optimal_leverage =find_optimal_leverage(some_returns)
        max_withdraw = find_max_safe_rate_to_withdraw(some_returns=some_returns, optimal_leverage=optimal_leverage, min_value_of_final_value=min_value_of_final_value)

        results_this_sr.append(max_withdraw)

    all_results[sr] = results_this_sr

q_results = {}
for quantile_point in [.1, .2, .3, .5, .75]:
    q_results_this_quantile = {}
    for sr in sr_options:
        point_value = pd.Series(all_results[sr]).quantile(quantile_point)
        q_results_this_quantile[sr] = point_value

    q_results[quantile_point] = q_results_this_quantile


q_to_plot =pd.DataFrame(q_results)

