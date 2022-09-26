#date: 2022-09-26T17:27:44Z
#url: https://api.github.com/gists/9ff22be210267a6d8cb48ff920b67bdf
#owner: https://api.github.com/users/Rafaelleafar

records = [
    ('foo',1,2),
    ('bar','hello'),
    ('foo',3,4),
]
def do_foo(x,y):
    print('foo',x,y)

def do_bar(s):
    print('bar',s)

for tag, *args in records:
    if  tag == 'foo':
        do_foo(*args)
    elif tag == 'bar':
        do_bar(*args)
