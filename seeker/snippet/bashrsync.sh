#date: 2025-07-18T17:03:23Z
#url: https://api.github.com/gists/348b1eab28564b71f9fe6171d17552e4
#owner: https://api.github.com/users/pedroamador

#!/bin/bash

# --- Configuración ---
ORIGEN="/ruta/a/origen"
DESTINO="/ruta/a/destino"
MARCA_TIEMPO="$DESTINO/marca_tiempo"

# --- Lógica del Script ---

# 1. Asegurarse de que el directorio de destino existe
mkdir -p "$DESTINO"

# 2. Comprobar si es la primera copia (si no existe el archivo de marca de tiempo)
if [ ! -f "$MARCA_TIEMPO" ]; then
  echo "--- Primera copia: Copiando todos los archivos... ---"
  # Copia todo el contenido preservando la estructura
  cp -a "$ORIGEN/." "$DESTINO/"

else
  echo "--- Copia incremental: Buscando archivos modificados... ---"
  # Busca archivos en ORIGEN más nuevos que la MARCA_TIEMPO y los copia
  # El comando -exec cp ... {} + es más eficiente que -exec cp ... \;
  find "$ORIGEN" -newer "$MARCA_TIEMPO" -exec cp -a -v --parents {} "$DESTINO/" \;

fi

# 3. Si la copia fue exitosa, actualiza la marca de tiempo para la próxima vez
if [ $? -eq 0 ]; then
  touch "$MARCA_TIEMPO"
  echo "--- Copia completada y marca de tiempo actualizada. ---"
else
  echo "--- Error durante la copia. La marca de tiempo no se ha actualizado. ---"
fi
