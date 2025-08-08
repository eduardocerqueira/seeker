#date: 2025-08-08T17:09:35Z
#url: https://api.github.com/gists/e1d91f94387ec3aff55fd0b5bb91c190
#owner: https://api.github.com/users/datavudeja

import pandas as pd
import attr
from factory import Factory, Sequence
from factory.fuzzy import FuzzyChoice, FuzzyInteger

class ConceptFactory(Factory):
   class Meta:
      model = attr.make_class("Concept", ['name', 'docdate', 'contextdate', 'concept', 'value'])
   name = Sequence(lambda x:'company {}'.format(x))
   docdate = 2017
   contextdate = 2017
   concept = FuzzyChoice(choices=['conceptA', 'conceptB', 'conceptC', 'conceptD', 'conceptE'])
   value = FuzzyInteger(low=0, high=100)

names = ['companyA', 'companyB', 'companyC']
years = [2017, 2016, 2015, 2014]
concepts = []
for name in names:
    for year in years:
        concepts += ConceptFactory.build_batch(size=4, name=name, docdate=year, contextdate=year)
        concepts += ConceptFactory.build_batch(size=4, name=name, docdate=year, contextdate=year-1)
        concepts += ConceptFactory.build_batch(size=4, name=name, docdate=year, contextdate=year-2)
df = pd.DataFrame([attr.asdict(c) for c in concepts])
df = df.sort_values(['name', 'docdate', 'contextdate'], ascending=[True, False, False])

from collections import defaultdict
dt1 = defaultdict(set)
dt2 = []
for name, group in df.groupby('name'):
    for idx, row in group.iterrows():
        docdate = row['docdate']
        contextdate = row['contextdate']
        if contextdate not in dt1[name]:
            dt1[name].add(contextdate)
            dt2.append({
                'name': name,
                'docdate': docdate,
                'contextdate': contextdate,
            })
df2 = pd.DataFrame(dt2)
df3 = pd.merge(df, df2, on=['name', 'docdate', 'contextdate'])

total_groups = len(df3.groupby(['name', 'docdate', 'contextdate']))
df4 = df3[df3.concept.isin(['conceptA', 'conceptB', 'conceptC'])]
df4.groupby('concept', as_index=False).name.aggregate(lambda x: len(x)/total_groups).rename(columns={'name':'pct'})
"""
    concept       pct
0  conceptA  0.666667
1  conceptB  1.055556  <- need fix. same context appear multiple times in the same name,docdate,contextdate group
2  conceptC  0.611111
"""


"""
output multiple calculations per group.
Each group df needs to be reduced into a single number for each func
"""
df.groupby('c').apply(lambda x: pd.Series({'func1': 10, 'func2': 20}, name='funcs'))
"""
funcs  func1  func2
c
3         10     20
9         10     20
"""

"""
counting concepts
"""
df = pd.DataFrame({
   'a':['foo', 'foo', 'foo', 'bar', 'bar', 'bar'], 
   'b':['bee', 'bee', 'boo', 'boo', 'bee', 'baz'], 
   'c':list('123456')
})

def count_concept(group_df):
   data = {'bee':0, 'boo':0, 'baz':0}
   data.update(group_df.b.value_counts().to_dict())
   return pd.Series(data)

df.groupby('a').apply(count_concept)
"""
     baz  bee  boo
a
bar    1    1    1
foo    0    2    1
"""