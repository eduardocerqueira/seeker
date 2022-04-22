#date: 2022-04-22T17:01:42Z
#url: https://api.github.com/gists/c8df207f64c34d050d24f353398a3fbb
#owner: https://api.github.com/users/ifrankandrade

# OK
for i, country in enumerate(countries):
    population = populations[i]
    print(f'{country} has a population of {population} million people')

# Much Better
for country, population in zip(countries, populations):
    print(f'{country} has a population of {population} million people')