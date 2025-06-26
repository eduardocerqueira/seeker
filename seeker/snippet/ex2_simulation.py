#date: 2025-06-26T17:01:35Z
#url: https://api.github.com/gists/2db5b36eacf9683d729e231ae811a50f
#owner: https://api.github.com/users/LSzubelak


seasonality = np.random.normal(0, 1, n)
economy = np.random.normal(0, 1, n)
ad_spend = 0.6 * seasonality + 0.4 * economy + np.random.normal(0, 1, n)
sales = 3.0 * ad_spend + 1.0 * seasonality + 0.5 * economy + np.random.normal(0, 1, n)

data2 = pd.DataFrame({
    'seasonality': seasonality,
    'economy': economy,
    'ad_spend': ad_spend,
    'sales': sales
})