#date: 2022-04-28T17:12:49Z
#url: https://api.github.com/gists/1458aed2aaf326797ff3039073411428
#owner: https://api.github.com/users/mecevit

templates = [
              ["[prop1] of [nns]","SELECT [prop1] FROM [nns]"],
              ["[agg] [prop1] for each [breakdown]","SELECT [agg]([prop1]) , [breakdown] FROM [prop1] GROUP BY [breakdown]"],
              ["[prop1] of [nns] by [breakdown]","SELECT [prop1] , [breakdown] FROM [nns] GROUP BY [breakdown]"],
              ["[prop1] of [nns] in [location] by [breakdown]","SELECT [prop1] , [breakdown] FROM [nns] WHERE location = '[location]' GROUP BY [breakdown]"],
              ["[nns] having [prop1] between [number1] and [number2]","SELECT name FROM [nns] WHERE [prop1] > [number1] and [prop1] < [number2]"],
              ["[prop] by [breakdown]","SELECT name , [breakdown] FROM [prop] GROUP BY [breakdown]"],
              ["[agg] of [prop1] of [nn]","SELECT [agg]([prop1]) FROM [nn]"],
              ["[prop1] of [nns] before [year]","SELECT [prop1] FROM [nns] WHERE date < [year]"],
              ["[prop1] of [nns] after [year] in [location]","SELECT [prop1] FROM [nns] WHERE date > [year] AND location='[location]'"],
              ["[nns] [verb] after [year] in [location]","SELECT name FROM [nns] WHERE location = '[location]' AND date > [year]"],
              ["[nns] having [prop1] between [number1] and [number2] by [breakdown]","SELECT name , [breakdown] FROM [nns] WHERE [prop1] < [number1] AND [prop1] > [number2] GROUP BY [breakdown]"],
              ["[nns] with a [prop1] of maximum [number1] by their [breakdown]","SELECT name , [breakdown] FROM [nns] WHERE [prop1] <= [number1] GROUP BY [breakdown]"],
              ["[prop1] and [prop2] of [nns] since [year]","SELECT [prop1] , [prop2] FROM [nns] WHERE date > [year]"],
              ["[nns] which have both [prop1] and [prop2]","SELECT name FROM [nns] WHERE [prop1] IS true AND [prop2] IS true"],
              ["Top [number1] [nns] by [prop1]","SELECT name FROM [nns] ORDER BY [prop1] DESC LIMIT [number1]"]
]
template = random.choice(templates)
print("Sample Query Template  :", template[0])
print("SQL Translation        :", template[1])