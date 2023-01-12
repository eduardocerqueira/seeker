#date: 2023-01-12T16:57:22Z
#url: https://api.github.com/gists/f2d2be79f78beb800fba1c4018086909
#owner: https://api.github.com/users/dennisbakhuis

df = pd.read_parquet('../data/image_values.parquet')
gps = pd.read_parquet('../data/mystery_box/gps_data.parquet')

# Make time the index.
df = (
    df
    .dropna(subset='filtered')
    .drop_duplicates(subset='time')
    .set_index('time')
)

# Make sure that gps is withing the time limits of our image data.
gps = (
    gps
    .loc[gps.time > df.index.min()]
    .loc[gps.time < df.index.max()]
)

# Add the A value to the GPS dataset.
def get_distance(time):
    return df.iloc[df.index.get_indexer([time], method='nearest')[0]].filtered

gps['A'] = gps.time.apply(get_distance).astype(int)

gps.to_parquet('../data/mystery_box_dataset.parquet', index=False)