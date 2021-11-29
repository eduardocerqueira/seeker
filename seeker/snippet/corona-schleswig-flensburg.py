#date: 2021-11-29T16:57:07Z
#url: https://api.github.com/gists/181646e5bc0b00609ef6330a0080ddf8
#owner: https://api.github.com/users/jze

# Beispiel, wie man aus den Corona-Zahlen des Kreises 
# Schleswig-Flensburg http://opendata.schleswig-holstein.de/dataset/corona-zahlen-schleswig-flensburg 
# Diagramme zeichnen kann

import pandas as pd
df = pd.read_csv("https://opendatarepo.lsh.uni-kiel.de/data/schleswig-flensburg/corona.csv")
df["Datum"] = pd.to_datetime(df["Datum"], format='%d.%m.%Y')
df.sample(5)

# Diagramm für ein Amt
df[df['Amt/Gemeinde']=='Amt Südangeln'].set_index("Datum")[['Positiv Getestete','Aktive Quarantänen']].plot()

# ins Wide-Format umbauen
df_wide = df.pivot(index='Datum', columns='Amt/Gemeinde', values='Positiv Getestete')
df_wide.sample(5)

# Diagramm mit Linien für alle Ämter/Gemeinden
df_wide.plot(figsize=(16,8))