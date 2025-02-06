#date: 2025-02-06T16:59:15Z
#url: https://api.github.com/gists/6bcb87b5bed39cb5e9ce5bf85bc68e46
#owner: https://api.github.com/users/edwardc29

#!/bin/bash

# Encuentra los módulos del proyecto (directorios que contienen un build.gradle.kts o build.gradle)
MODULES=$(find . -mindepth 2 -type f \( -name "build.gradle.kts" -o -name "build.gradle" \) | sed 's|/build.gradle.*||' | sed 's|^\./||' | sort | uniq)

# Archivo donde se guardarán las dependencias
OUTPUT_FILE="all_dependencies.txt"
echo "📦 Listado de dependencias por módulo:" > "$OUTPUT_FILE"

echo "📦 Módulos encontrados:"
echo "$MODULES"
echo ""

# Itera sobre cada módulo y obtiene sus dependencias
for MODULE in $MODULES; do
    MODULE_NAME=$(echo "$MODULE" | tr '/' ':')  # Convierte la ruta en nombre de módulo Gradle
    echo "📌 Obteniendo dependencias para el módulo: $MODULE_NAME"

    # Escribe el encabezado del módulo en el archivo de salida
    echo "" >> "$OUTPUT_FILE"
    echo "🔹 Módulo: $MODULE_NAME" >> "$OUTPUT_FILE"
    echo "--------------------------------------" >> "$OUTPUT_FILE"

    # Ejecuta Gradle y filtra solo las dependencias directas
    ./gradlew "$MODULE_NAME:dependencies" --configuration debugCompileClasspath | grep -E "^[+|\\|]--- " | awk '{print $2}' >> "$OUTPUT_FILE"

    # Asegura que Gradle finaliza antes de continuar con el siguiente módulo
    wait
done

echo "✅ Dependencias extraídas en: $OUTPUT_FILE"