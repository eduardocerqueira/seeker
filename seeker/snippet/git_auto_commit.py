#date: 2025-04-18T16:35:20Z
#url: https://api.github.com/gists/8f17b39bb8522a26f7e6535031b1e9c7
#owner: https://api.github.com/users/mukundhan94

from git import Git
import os
import time
import schedule
path_to_dir='C:/Users/gopiprasanthp/Desktop/she'
git =Git(path_to_dir)

def commit():
    f=open(os.path.join(path_to_dir,'README.md'),'a+')

    f.write("\n New Line" + time.ctime())
    f.close()
    try:
        git.execute("git add README.md")
        commit_msg='"updated README"'
        git.execute("git commit -m "+commit_msg)
        git.execute("git push origin master")
    except Exception as exp:
        print(exp)

schedule.every(10).minutes.do(commit)

while True:
   schedule.run_pending()
   time.sleep(1)