#date: 2022-09-16T21:31:05Z
#url: https://api.github.com/gists/2d1d151aaf9652fe81b9b68be7405ce7
#owner: https://api.github.com/users/under0tech

# Add predicted results to the table
date_now = dt.date.today()
date_tomorrow = dt.date.today() + dt.timedelta(days=1)
date_after_tomorrow = dt.date.today() + dt.timedelta(days=2)

init_df.loc[date_now] = [predictions[0], f'{date_now}', 0.5]
init_df.loc[date_tomorrow] = [predictions[1], f'{date_tomorrow}', 0.6]
init_df.loc[date_after_tomorrow] = [predictions[2], f'{date_after_tomorrow}', 0.7]

# Result graphic
plt.style.use(style='ggplot')
plt.figure(figsize=(16,10))
plt.plot(init_df['close'][-150:].head(147))
plt.plot(init_df['close'][-150:].tail(4))
plt.xlabel('days')
plt.ylabel('price')
plt.legend([f'Actual price for {STOCK}', f'Predicted price for {STOCK}'])
plt.show()