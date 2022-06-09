#date: 2022-06-09T17:09:29Z
#url: https://api.github.com/gists/1a22824d4457febb14262616a6f7a412
#owner: https://api.github.com/users/StephenFordham

e1 = Employees(46347832, 30, 'Bournemouth')

for key, value in e1.__dict__.items():
    print('{}: {}'.format(key, value))
    
#Console output 

TypeError: Only valid attributes of type string are accepted