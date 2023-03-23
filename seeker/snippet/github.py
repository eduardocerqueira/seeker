#date: 2023-03-23T16:46:11Z
#url: https://api.github.com/gists/685cce869efd12dfce1fd2b1f1bcdc43
#owner: https://api.github.com/users/KruinKop

git config --global --list
git config --global user.name "jouw naam" # verplicht
git config --global user.email "jouw GitHub (!) email adres" # verplicht $ git config --global init.defaultBranch main
git config --global core.editor "code --wait"

git init #repository initialiseren
git status

git add . # alle wijzigingen toevoegen aan de staging area
git add main.py # een specifiek bestand toevoegen aan de staging area

git commit -m "Eerste commit - start van het project"
git push -u origin main

git remote add origin git@github.com:jouw_gebruikersnaam/jouw_reponaam.git
git remote remove origin

git clone https://github.com/jouw_gebruikersnaam/jouw_reponaam.git

git pull origin main

echo "venv" > .gitignore

git log # overzicht van alle commits
git log --oneline # heel kort overzicht van alle commits (handig!) 
git show <checksum> # wat is er gebeurd in die specifieke commit 
git log -p # wat is er gebeurd in alle commits
git log --graph # visualisatie
git log --grep="zoek regex" # op zoek naar een specifieke commit

git diff # tussen de werkmap en de laatste commit
git diff <commit> # tussen de werkmap en een specifieke commit 
git diff <commit1> <commit2> # tussen twee commits
git diff --staged # tussen de staging area en de laatste commit

git restore --staged mijn_bestand.txt 
git restore --staged .
git restore .

git log --oneline # commit-id kopiëren
git checkout <commit id=""> # je kan KIJKEN naar de oudere versie, niet wijzigen 
git checkout main # terugkeren naar de laatste versie

git log --oneline # commit-id kopiëren
git revert <commit> # maakt nieuwe commit aan met oude commit als nieuwe situatie