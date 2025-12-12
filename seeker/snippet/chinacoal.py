#date: 2025-12-12T17:16:50Z
#url: https://api.github.com/gists/fdcaf9511725016b28b197fa36ea0284
#owner: https://api.github.com/users/cavedave

#get all ember energy data
import pandas as pd

df = pd.read_csv("https://storage.googleapis.com/emb-prod-bkt-publicdata/public-downloads/monthly_full_release_long_format.csv",low_memory=False)

#get coal usage

import pandas as pd

df_china_coal_twh = df[
    (df['Area'] == 'China') &
    (df['Variable'] == 'Coal') &
    (df['Unit'] == 'TWh')
].copy()

df_china_coal_twh['Year'] = pd.to_datetime(df_china_coal_twh['Date']).dt.year

yearly_china_coal_generation = df_china_coal_twh.groupby('Year')['Value'].sum().reset_index()

display(yearly_china_coal_generation)

#get total power usage
df_china_total_twh = df[
    (df['Area'] == 'China') &
    (df['Variable'] == 'Total Generation') &
    (df['Unit'] == 'TWh')
].copy()

df_china_total_twh['Year'] = pd.to_datetime(df_china_total_twh['Date']).dt.year

yearly_china_total_generation = df_china_total_twh.groupby('Year')['Value'].sum().reset_index()

print("Yearly China Coal Generation:")
display(yearly_china_coal_generation)

print("\nYearly China Total Generation:")
display(yearly_china_total_generation)

#get percentage of total that is coal

yearly_china_coal_generation_renamed = yearly_china_coal_generation.rename(columns={'Value': 'Coal_Generation'})
yearly_china_total_generation_renamed = yearly_china_total_generation.rename(columns={'Value': 'Total_Generation'})

merged_df = pd.merge(
    yearly_china_coal_generation_renamed,
    yearly_china_total_generation_renamed,
    on='Year',
    how='inner'
)

merged_df['Percentage_Coal'] = (merged_df['Coal_Generation'] / merged_df['Total_Generation']) * 100

display(merged_df)

#graph it

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(merged_df['Year'], merged_df['Percentage_Coal'], marker='o', linestyle='-', color='black')

plt.title("China's Yearly Coal Percentage in Total Electricity Generation", fontsize=16)
plt.xlabel('Year')
plt.ylabel('Percentage of Total Generation from Coal (%)')

plt.grid(True, linestyle='--', alpha=0.6)

plt.xticks(merged_df['Year'].unique(), rotation=45)

plt.tight_layout()
plt.show()

#do the same for total coal usage (for electricity)

# Filter for relevant years (e.g., 2022 onwards, or all years where 10 months are available)
start_year = 2015 # Changed start year to include more historical context
df_china_coal_multi_year = df_china_coal_twh[df_china_coal_twh['Year'] >= start_year].copy()

# Ensure 'Date' column is datetime and extract month
df_china_coal_multi_year['Date'] = pd.to_datetime(df_china_coal_multi_year['Date'])
df_china_coal_multi_year['Month'] = df_china_coal_multi_year['Date'].dt.month

# Filter for the first 10 months (January to October)
df_china_coal_10_months = df_china_coal_multi_year[df_china_coal_multi_year['Month'] <= 10].copy()

# Group by Year and sum the 'Value' for these 10 months
yearly_china_coal_10_months = df_china_coal_10_months.groupby('Year')['Value'].sum().reset_index()

display(yearly_china_coal_10_months)

# Plotting the data
plt.figure(figsize=(12, 6))
plt.plot(yearly_china_coal_10_months['Year'], yearly_china_coal_10_months['Value'],
         marker='o', linestyle='-', color='black', label='China Coal Generation')

plt.title('China Monthly Coal Electricity Generation (Jan-Oct)', fontsize=16)
#plt.suptitle('First 10 Months of Each Year', fontsize=12, y=0.92)
plt.xlabel('Year')
plt.ylabel('Annual Generation (TWh)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(yearly_china_coal_10_months['Year'].unique(), rotation=45) # Set x-ticks to show all years
plt.legend()
plt.tight_layout()
plt.show()