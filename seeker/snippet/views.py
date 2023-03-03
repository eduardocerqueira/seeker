#date: 2023-03-03T16:47:45Z
#url: https://api.github.com/gists/5b38235e701a9e4d26feaef35c20cba2
#owner: https://api.github.com/users/docsallover

from django.shortcuts import render

def my_view(request):
    context = {'name': 'John'}
    return render(request, 'my_template.html', context)