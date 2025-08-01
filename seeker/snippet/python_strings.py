#date: 2025-08-01T17:08:16Z
#url: https://api.github.com/gists/c4f79ca5d9da02fc68d78ccba55ab0ad
#owner: https://api.github.com/users/ocerino

highlighted_poems = "Afterimages:Audre Lorde:1997,  The Shadow:William Carlos Williams:1915, Ecstasy:Gabriela Mistral:1925,   Georgia Dusk:Jean Toomer:1923,   Parting Before Daybreak:An Qi:2014, The Untold Want:Walt Whitman:1871, Mr. Grumpledump's Song:Shel Silverstein:2004, Angel Sound Mexico City:Carmen Boullosa:2013, In Love:Kamala Suraiyya:1965, Dream Variations:Langston Hughes:1994, Dreamwood:Adrienne Rich:1987"

print(highlighted_poems)

highlighted_poems_list = highlighted_poems.split(",")
print(highlighted_poems_list)

highlighted_poems_stripped = [s.strip() for s in highlighted_poems_list]
print(highlighted_poems_stripped)

highlighted_poems_details = [s.split(":") for s in highlighted_poems_stripped]
print(highlighted_poems_details)

titles = [s[0] for s in highlighted_poems_details]
poets = [s[1] for s in highlighted_poems_details]
dates = [s[2] for s in highlighted_poems_details]

for i in range(len(titles)):
  print("The poem {title} was published by {poet} in {date}".format(title=titles[i], poet=poets[i], date=dates[i]))