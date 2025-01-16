#date: 2025-01-16T17:08:36Z
#url: https://api.github.com/gists/d76951e03e1dd69e58e1fd4e28c95936
#owner: https://api.github.com/users/zapaiamarce

# Esta secuencia de comandos sirve cuando tenemos un fork
# clonado en nuestro compu y queremos traernos cambios
# del repo original (el que forkeamos)

# Cuando clonamos un repo de GitHub, nuestro repo local
# queda enganchado al repo remoto para que podamos hacer pull y push
# Apenas clonamos se va a generar un remoto llamado "origin"

# Para hacer pull y push de otro repo remoto debemos agregar un nuevo remoto
# en este caso lo voy a llamar "repo-original"
git remote add repo-original git@github.com:apx-school/forks-y-prs.git

# Una vez que tengo el repo original como remoto puedo hacer pull
# en este caso aclaro el branch también 
git pull repo-origin main