#date: 2021-09-01T16:55:35Z
#url: https://api.github.com/gists/91b672b7a5e96c71bf9341a6e442ffea
#owner: https://api.github.com/users/EnrageStudio

#!/bin/bash

Old=$'Parametro actual'
New=$'Parametro nuevo'
Path=(Directorio de los portales)
Drupal="settings.php"
Wp="wp-config.php"
find $Path -type f -name $Drupal -exec sed -i "s/$Old/$New/" {} +
echo "Completo cambio en Drupal"
find $Path -type f -name $Wp -exec sed -i "s/$Old/$New/" {} +
echo "Completo cambio wordpress"
#Limpieza de variables
unset Old
unset New
unset Path
unset Drupal
unset Wp

echo "Tarea completada"
exit