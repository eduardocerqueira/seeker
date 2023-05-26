#date: 2023-05-26T17:06:23Z
#url: https://api.github.com/gists/9fcfacaaa9e0702e43c7ff0c6004398f
#owner: https://api.github.com/users/gffx

import os
folder=f'{os.getcwd()}\\css'
style=icon=script=nav=meta=jsc='';links=[]
if not os.path.exists(folder):os.makedirs(folder)
try:os.path.getsize(os.getcwd()+'\\home.txt')
except:open(f'{os.getcwd()}\\home.txt','w').write('$title:my website\nhere is some content!!!\nwrite "$title:YOUR TITLE HERE" on the first line to rename the website')

if not os.path.isfile(f'{os.getcwd()}\\home.txt'):open(f'{os.getcwd()}\\home.txt','w')
for fn in os.listdir(os.getcwd()):
    try:fname=os.path.splitext(fn)[0]
    except:continue
    if fn.endswith('.ico'):icon=f'<link rel="icon"href="{fn}">\n'
    elif fn.endswith('.css'):meta+=f'<link rel="stylesheet"href="{fn}">\n'
    elif fn.endswith('.js'):jsc+=f'<script src="{fn}"></script>\n'
    elif fn.endswith('.txt')and fn!='home.txt':nav+=f'<a href="{folder}\\{fname}.html">{fname}</a>\n'
    if fn=='home.txt':
        lines = open(fn,'r').readlines()
        if lines[0].startswith('$title:'):meta+=f'<h1><a href="index.html">{lines[0][7:-1]}</a></h1>\n';ttl=lines[0][7:-1]
for fn in os.listdir(os.getcwd()):
    try:fname=os.path.splitext(fn)[0]
    except:continue
    if fn.endswith('.txt'):
        if not fname=='home':open(f'{folder}\\{fname}.html','w').write(f'<title>{ttl} - {fname}</title><meta charset="UTF-8">\n{icon}{meta}<nav>\n{nav}\n</nav>\n<div><p>'+open(f'{os.getcwd()}\\{fn}','r').read())+jsc
        else:open(f'{folder}\\index.html','w').writelines(f'<title>{ttl}</title><meta charset="UTF-8">\n{icon}{meta}<nav>\n{nav}\n</nav>\n<div><p>'+str(''.join(open(fn,'r').readlines()[1:])))
    if fn.endswith(('.css','.js')):open(folder+'\\'+fn,'w').write(open(os.getcwd()+'\\'+fn,'r').read())
    elif fn not in('.git','errors.txt','render.py'):
        while True:
            try:open(os.getcwd()+'\\'+folder+'\\'+fn,'wb').write(open(os.getcwd()+'\\'+fn,'rb').read(4096))
            except:break