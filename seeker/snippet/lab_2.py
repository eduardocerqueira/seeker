#date: 2023-11-24T16:55:45Z
#url: https://api.github.com/gists/94a38f47f2fbf6f0092a42a8ae4823d4
#owner: https://api.github.com/users/Mishganio

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('planets.csv') 

#1.Почистите выборку:
#удалить планеты с planet_type="Unknown"
df = df[df['planet_type']!= "Unknown"]

#удалить планеты с незаполненной массой (столбцы mass_multiplier и mass_wrt, аббревиатура расшифровывается как with respect to).
df = df.query('mass_multiplier.notnull() and mass_wrt.notnull()')

#удалить планеты с незаполненным радиусом (столбцы radius_multiplier и radius_wrt)
df = df.query('radius_multiplier.notnull() and radius_wrt.notnull()')

#удалить планеты с незаполненным расстоянием от Земли (столбец distance)
df = df.query('distance.notnull()')

#2.Для каждой планеты посчитайте радиус в тысячах километров.
#Для этого нужно умножить значение radius_multiplier на радиус планеты, указанной в поле radius_wrt и поделить на 1000.
#Для справки: радиус Юпитера составляет 69911 км, радиус Земли 6371 км.

df.loc[df['radius_wrt']=='Jupiter','radius'] = df['radius_multiplier']*69911/1000
df.loc[df['radius_wrt']=='Earth','radius'] = df['radius_multiplier']*6371/1000

#3.Для каждой планеты посчитайте массу в килограммах. Алгоритм аналогичен п. 2.
#Для справки: масса Юпитера 1.89 * 10^27 кг, масса Земли 5.97 * 10^24 кг

df.loc[df['mass_wrt']=='Jupiter','mass'] = df['mass_multiplier']*1.89e27
df.loc[df['mass_wrt']=='Earth','mass'] = df['mass_multiplier']*5.97e24

#4.Для каждого типа планеты посчитайте:
#наименьшее и наибольшее расстояние от Земли (в датасете оно выражено в световых годах),
#среднюю массу,
#средний радиус,
#количество в выборке

agg_result = df.groupby('planet_type').agg({'distance':['min','max'],'mass':'mean','radius':'mean','name':'count'})

#5.Постройте круговую гистограмму (pie chart) количества планет каждого типа. Сохраните изображение в файл

agg_result['name']['count'].plot(kind='pie',label='')
plt.savefig('result.png')   

