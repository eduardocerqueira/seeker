#date: 2021-09-09T17:05:00Z
#url: https://api.github.com/gists/40d5a0d74044587d1d86d5de8973374d
#owner: https://api.github.com/users/ArgueraDev

#!/bin/sh

#.............................................
#       Roberto Carlos Arguera Campos
#.............................................

#Este script es para crear grupos de usuario y sus respectivos usuarios...
#Como primer paso es necesario haber iniciado como root
#crearemos un usuario nuevo
useradd carlos
#a continuacion sera necesario agregar una contraseña y otros datos que pedira
#para verificar si el usuario a sido creado utilizamos el siguiente comando
cat /etc/passwd
#ahora comenzaremos creando el grupo con el siguiente comando
groupadd casa
#Si desea verificar si se creo el grupo podemos utilizar el siguiente comando
cat /etc/group #este nos mostrara un listado de los grupos creados y al final apareceza nuestro grupo

#ahora procederemos a agregar dos usuarios dentro del grupo que acabamos de crear
#para esto utilizaremos en siguiente comando
usermod -a -G casa carlos #el usuario que hemos creado en esta sesion
usermod -a -G casa roberto #el usuario que se creo con el sistema
#para visualizar si se agregaron los usuarios al grupo utilizamos otra vez el siguiente comando
cat /etc/group
#a la par del nombre del grupo nos apareceran los nombres de los usuarios
#por ultimo realizaremos una modificacion al nombre del grupo con el siguiente comando 
groupmod -g 1001 -n Familia casa
#tenemos que buscar la gid de nuestro grupo que queremos modificar y luego el nombre nuevo y por utlimo el nombre actual
#para verificar el cambio podemos utilizar nuevamente el siguiente comando
cat /etc/group

#este seria mi practica


