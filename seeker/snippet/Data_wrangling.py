#date: 2023-01-30T16:53:58Z
#url: https://api.github.com/gists/aebbbbb945cadace41f633726d822bd2
#owner: https://api.github.com/users/captn-jack-sparrow

import pandas as pd
import io
import matplotlib.pyplot as plt

#This fetches the cast lists
fole = open("C:/Users/Obama/Downloads/castfile.csv/cast.csv")
cast_lists = fole.read()

#This fetches the release dates
sll = open("C:/Users/Obama/Downloads/castfile.csv/release_dates.csv")
release_dates = sll.read()

#This fetches the titles
tii = open("C:/Users/Obama/Downloads/Castfile.csv/titles.csv")
titles = tii.read()

# creating the dfs

cast_list = pd.read_csv(io.StringIO(cast_lists))
release_date = pd.read_csv(io.StringIO(release_dates))
titless = pd.read_csv(io.StringIO(titles))

# Section 1 Q1

harryp = titless[titless.title.str.contains("harry potter", case=False)]
harryp = harryp.sort_values(["year"], ascending=False)
print(harryp)

# Section 1 Q2

released_2015 = titless[titless.year == 2015]
print(released_2015)

# Section 1 Q3

movies_between_2000_2018 = titless[(titless.year >= 2000) & (titless.year <= 2018)]
print(movies_between_2000_2018)

# Section 1 Q4

ham_count = titless[titless.title=="Hamlet"]
print(ham_count)
ham_count_final = len(ham_count)
print(ham_count_final)

# Section 1 Q5

recent_ham = ham_count[ham_count.year >= 2000]
print(recent_ham)

# Section 1 Q6

inception = cast_list[(cast_list.title == "Inception") & (cast_list.n.isna())]
print(inception)

# Section 1 Q7

inception_main = cast_list[(cast_list.title == "Inception") & (cast_list.n.notna())]
print(len(inception_main))

# Section 1 Q8

inception_main = inception_main.sort_values("n")

print(inception_main.head(10))

# Section 1 Q9
# part a

al_search = cast_list[cast_list.character == "Albus Dumbledore"]
al_movies = al_search["title"]
print(al_movies)

# part b

al_unique = al_search["name"]
al_unique = al_unique.drop_duplicates()

print(al_unique)


# Section 1 Q10
# part a

kr = cast_list[cast_list.name == "Keanu Reeves"]
print(len(kr))

# past b

kr["leading_roles"] = kr.n.notna()
kror = kr[(kr.leading_roles == True) & (kr.year >=1999)]
kror = kror.sort_values("year")
print(kror)

# Section 1 Q11
# part a

roles_56 = cast_list[(cast_list.year >= 1950) & (cast_list.year <= 1960)]
roles_a56 = len(roles_56)
print(roles_a56)

# part b

roles_0717 = cast_list[(cast_list.year >= 2007) & (cast_list.year <=2017)]
roles_a0717 = len(roles_0717)

print(roles_a0717)

# Section 1 Q12
# part a

dfroles = cast_list[(cast_list["year"] >= 2000) & (cast_list["n"].notna())]
rlc = len(dfroles)
main_rdf = dfroles[dfroles.n <= 10]
print(main_rdf)

# part b

rlc = len(dfroles)
print(rlc)



# part c

dfrolesn = cast_list[(cast_list.year >= 2000) & (cast_list.n.isna())]

nrlc = len(dfrolesn)
print(nrlc)

# Section 2 Q1

titles2 = titless[titless.year >= 2000]

year_counti = titles2.year.value_counts()

print(year_counti.head(3))

# Section 2 Q2

titless["decade"] = (round(titless.year/10)*10)

deccount = titless.decade.value_counts()

deccount.plot(kind="bar")

plt.show()


# Section 2 Q3
# part a

poppin_names = cast_list.character.value_counts()[:10]

print(poppin_names)
print(cast_list.columns)
# part b

s = cast_list[(cast_list["character"] == "Himself") | (cast_list["character"] == "himself")]


print(s)
himmscount = s.name.value_counts()
print(himmscount.head(10))


s = cast_list[(cast_list["character"] == "Herself") | (cast_list["character"] == "herself")]

herscount = s.name.value_counts()
print(herscount.head(10))

# Section 2 Q4
# part a
zom_char = cast_list[cast_list.character.str.contains("Zombie")]

zom_count = zom_char.character.value_counts()

print(zom_count.head(10))

# part b

zom_char = cast_list[cast_list.character.str.contains("Police")]

zom_count = zom_char.character.value_counts()

print(zom_count.head(10))

# Section 2 Q5

k_r = cast_list[cast_list["name"] == "Keanu Reeves"]
year_count = k_r.year.value_counts()
year_count.sort_index(inplace = True)
year_count.plot(kind="bar")

plt.show()

# Section 2 Q6

plt.scatter(k_r["year"], k_r["n"])

plt.show()

# Section 2 Q7
print(titless.head())
ham_d = titless[titless.title.str.contains("hamlet", case=False)]

ham_d["decade"] = round(ham_d.year/10)*10
hh = ham_d.value_counts("decade")

hh.plot(kind="bar")
plt.show()

# Section 2 Q8
# part a
# the criteria I will use to decide whether or not a role is a "lead role" is the following,
# I only be looking at roles with an n value of 1

cast_list_60_69 = cast_list[(cast_list.year >= 1960) & (cast_list.year <= 1969)]
cast_list_60_69_lead = cast_list_60_69.n == 1
roles_available_60_69 = len(cast_list_60_69_lead)
print(roles_available_60_69)

# part b
# the same criteria is used below as above for determining "leading roles"

cast_list_00_09 = cast_list[(cast_list.year >= 2000) & (cast_list.year <= 2009)]
cast_list_00_09_lead = cast_list_00_09.n == 1
cast_lead_00_09 = len(cast_list_00_09_lead)
print(cast_lead_00_09)

# Section 2 Q9

print(cast_list.columns)

frank = cast_list[cast_list.name == "Frank Oz"]
franky = frank.title.value_counts()
fran = franky[franky >= 2]
fran_df = fran.to_frame()
fran_df = fran_df.reset_index()

fran_df = fran_df.rename(columns={'title': 'new_name'})
fran_df = fran_df.rename(columns={'index': 'title'})

m_df = pd.merge(cast_list, fran_df, on="title")
mmm = m_df[m_df.name == "Frank Oz"]
mmd = mmm[["year", "title"]].drop_duplicates()
mmmm = mmd.sort_values("year")
print(mmmm)

# Section 2 Q10

frankz = frank.character.value_counts()
deez = frankz[frankz >= 2]

print(deez)

#Section 3 Q1

# Section 3 Q1
print(release_date.columns)
print(release_date.date.dtypes)
release_date.date = pd.to_datetime(release_date.date)
release_date.title = release_date.title.astype(str)
christmas = release_date[(release_date.title.str.contains('Summer')) & (release_date.country == 'USA')]
christmas.date.dt.month.value_counts().sort_index().plot(kind='bar')
plt.show()

# Section 3 Q2

christmas = release_date[(release_date.title.str.contains('Action')) & (release_date.country == 'USA')]
christmas.date.dt.week.value_counts().sort_index().plot(kind='bar')
plt.show()

# Section 3 Q3
cast_list_kr = cast_list[(cast_list.name == "Keanu Reeves") & (cast_list.n == 1)]
release_date_us = release_date[release_date.country == "USA"]
krdf = pd.merge(release_date_us, cast_list_kr, on=["title", "year"], how="inner")

krdf_sorted = krdf.sort_values("year")
j = len(krdf)
print(krdf_sorted.head(j))


# Section 3 Q4

list_kr = cast_list[(cast_list.name == "Keanu Reeves")]
date_us = release_date[release_date.country == "USA"]
rdf = pd.merge(date_us, list_kr, on=["title", "year"], how="inner")
mr = rdf.date.dt.month.value_counts().sort_index().plot(kind="bar")
plt.show()

# Section 3 Q5

ian = cast_list[cast_list.name == "Ian McKellen"]
country = release_date[release_date.country == "USA"]
i_c = pd.merge(ian, country, on = ["title", "year"], how="inner")
i_c.year.value_counts().sort_values().plot()
plt.show()