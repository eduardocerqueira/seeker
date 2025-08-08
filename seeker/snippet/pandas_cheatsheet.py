#date: 2025-08-08T17:09:35Z
#url: https://api.github.com/gists/e1d91f94387ec3aff55fd0b5bb91c190
#owner: https://api.github.com/users/datavudeja

import pandas as pd
import numpy as np

"""
create dataframe with random data.
"""

"""
   A  B  C
0  9  8  0
1  6  7  3
2  9  5  3
"""
df = pd.DataFrame(np.random.randint(0,10,size=(3,3)), columns=list('ABC'))

"""
   employees       name rating      sector
0         99  company 0      C     finance
1         37  company 1      A      retail
2         65  company 2      B     finance
"""
import attr
from factory import Factory, Sequence
from factory.fuzzy import FuzzyChoice, FuzzyInteger
class CompanyFactory(Factory):
   class Meta:
      model = attr.make_class("Company", ['name', 'sector', 'rating', 'employee_count'])
   name = Sequence(lambda x:'company {}'.format(x))
   sector = FuzzyChoice(choices=['technology', 'healthcare', 'finance', 'retail', 'auto'])
   rating = FuzzyChoice(choices=['A', 'B', 'C'])
   employee_count = FuzzyInteger(low=0, high=100)
df = pd.DataFrame([attr.asdict(c) for c in CompanyFactory.build_batch(size=3)])


"""
Read/Write dataframe into csv file.
"""
df.to_csv('/tmp/test.csv', encoding='utf8') . # index will be written to csv as first column, without a column name
pd.read_csv('/tmp/test.csv', encoding='utf8', index_col=0) . # first column in csv will be used as the index

"""
printing dataframe with tabulate
"""
from tabulate import tabulate
print(tabulate(df[:5], headers='keys'))

"""
filtering dataframe with multiple WHERE clause
"""
is_technology_or_finance = df.sector.isin(['technology', 'finance'])
is_rating_A = df.rating.eq('A')
is_between_30_and_50_employees = df.employee_count.between(30,50)
df[np.logical_or.reduce([is_technology_or_finance, is_rating_A, is_between_30_and_50_employees])]

"""
GROUPBY AGGREGATE
"""

class ScoreFactory(Factory):
   class Meta:
      model = attr.make_class("Score", ['player', 'season', 'points'])
    player = FuzzyChoice(choices=['john', 'meg', 'peter', 'alan'])
    season = FuzzyChoice(choices=[2010, 2011, 2012])
    points = FuzzyInteger(low=1, high=10)

df = pd.DataFrame([attr.asdict(score) for score in ScoreFactory.build_batch(size=100)])
print(tabulate(df[:3], headers='keys'))
"""    
    player      points    season
--  --------  --------  --------
 0  john             8      2011
 1  meg              6      2010
 2  peter            4      2012
 3  meg             10      2010
 4  john             5      2010
 5  alan             5      2012
 6  alan             3      2011
 7  peter            3      2010
 8  peter            2      2010
 9  john             1      2011
"""

"""
GROUPBY AGGREGATE single column
"""
df.groupby(['player', 'season'], as_index=False).aggregate(np.sum)
"""
    player      season    points
--  --------  --------  --------
 0  alan          2010        37
 1  alan          2011        44
 2  alan          2012        64
 3  john          2010        31
 4  john          2011        75
 5  john          2012        41
 6  meg           2010        43
 7  meg           2011        18
 8  meg           2012        44
 9  peter         2010        44
10  peter         2011        45
11  peter         2012        47
"""

"""
Custom AGGREGATE function
"""
join_string = lambda x: x.apply(str).str.cat(sep=',')
df.groupby(['player', 'season'], as_index=False).aggregate(join_string)
"""
    player      season  points
--  --------  --------  -----------------------------
 0  alan          2010  1,6,2,2,4,2,6,8,6
 1  alan          2011  3,10,10,3,1,8,6,3
 2  alan          2012  5,8,7,10,8,8,1,1,9,7
 3  john          2010  5,8,2,10,1,5
 4  john          2011  8,1,1,4,8,10,1,6,4,6,9,1,6,10
 5  john          2012  9,6,7,5,1,2,9,2
 6  meg           2010  6,10,9,7,10,1
 7  meg           2011  6,1,5,6
 8  meg           2012  1,8,6,8,1,8,6,6
 9  peter         2010  3,2,3,7,1,2,10,3,8,2,3
10  peter         2011  1,8,1,4,10,8,5,8
11  peter         2012  4,7,9,5,8,8,5,1
"""

"""
Aggregate different columns with different functions
"""
join_string = lambda x: x.apply(str).str.cat(sep=',')
df.groupby('player', as_index=False).aggregate({'points': join_string, 'season': np.max})
"""
    player    points                                                        season
--  --------  ----------------------------------------------------------  --------
 0  alan      5,3,10,8,7,10,1,6,2,2,10,8,8,1,3,1,1,4,9,2,8,6,8,6,7,6,3        2012
 1  john      8,5,1,9,6,7,5,1,4,1,8,2,10,8,1,9,6,4,6,2,9,10,1,1,6,5,10,2      2012
 2  meg       6,10,1,8,6,9,1,6,8,1,7,10,8,5,6,6,6,1                           2012
 3  peter     4,3,2,3,7,1,1,8,2,10,1,4,3,8,10,7,9,5,2,8,5,8,8,8,5,1,3         2012
 """

"""
Aggregate multiple functions to same column
"""
df.groupby(['player', 'season'], as_index=False).aggregate({'points':{'sum_points':np.sum, 'max_points':np.max}})
"""
   player season     points
                 sum_points max_points
0    alan   2010         37          8
1    alan   2011         44         10
2    alan   2012         64         10
3    john   2010         31         10
4    john   2011         75         10
5    john   2012         41          9
6     meg   2010         43         10
7     meg   2011         18          6
8     meg   2012         44          8
9   peter   2010         44         10
10  peter   2011         45         10
11  peter   2012         47          9
"""

"""
Getting all scores for each player's best season
Approach 1: using pd.merge
"""
df1 = df.groupby(['player', 'season'], as_index=False).sum()
df2 = df1.groupby('player', as_index=False).points.max()
df3 = pd.merge(df1, df2, on=['player', 'points'])
pd.merge(df, df3, on=['player', 'season'], suffixes=('', '_season_total'))
"""
Approach 2: using idxmax
"""
df1 = df.groupby(['player', 'season'], as_index=False).sum()
idx = df1.groupby('player').points.idxmax()
df2 = df1.iloc[idx]
pd.merge(df, df2, on=['player', 'season'], suffixes=('', '_season_total'))
"""
Approach 3: using apply
"""
df1 = df.groupby(['player', 'season'], as_index=False).sum()
df2 = df1.groupby('player', as_index=False).apply(lambda t: t[t.points==t.points.max()])
pd.merge(df, df2, on=['player', 'season'], suffixes=('', '_season_total'))
