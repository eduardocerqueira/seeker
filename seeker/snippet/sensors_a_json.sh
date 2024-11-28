#date: 2024-11-28T16:52:50Z
#url: https://api.github.com/gists/95268ec9d2ee32aaf6c02cd6a1e561f7
#owner: https://api.github.com/users/arteze

#!/bin/bash

echo "{"
bloque=""
primer_bloque="1"
echo -e "$(sensors)\n" | while IFS= read -r linea; do
	if [[ -z "$linea" ]]; then
		if [[ -n "$bloque" ]]; then
			dispositivo="$(echo "$bloque" | head -n 1)"
			valor="$(echo "$bloque" | tail -n+2)"
			
			# Extraer y limpiar valores
			temp1=$(echo "$valor" | grep -oP 'temp1:\s+\+(\S+)' | sed 's/temp1:\s\+//')
			high=$(echo "$valor" | grep -oP 'high\s*=\s*(\S+)' | sed 's/^[^=]*=\s*//')
			crit=$(echo "$valor" | grep -oP 'crit\s*=\s*(\S+)' | sed 's/^[^=]*=\s*//')
			hyst=$(echo "$valor" | grep -oP 'hyst\s*=\s*(\S+)' | sed 's/^[^=]*=\s*//')
			adapter=$(echo "$valor" | grep -oP 'Adapter:\s+(\S+\s+\S+)' | sed 's/Adapter:\s\+//' | sed 's/[[:space:]]*$//')

			# Limpiar paréntesis y comas, solo al final de los valores
			high=$(echo "$high" | sed 's/[,)]]*$//')
			crit=$(echo "$crit" | sed 's/[,)]]*$//')
			hyst=$(echo "$hyst" | sed 's/[,)]]*$//')

			# Formatear el JSON para el dispositivo con sus subpropiedades
			if [[ "$primer_bloque" == "1" ]]; then
				echo " \"$dispositivo\": {"
				echo "   \"Adapter\": \"$adapter\","
				echo "   \"temp1\": \"$temp1\","
				echo "   \"high\": \"$high\","
				echo "   \"crit\": \"$crit\","
				echo "   \"hyst\": \"$hyst\""
				echo " }"
			else
				echo ", \"$dispositivo\": {"
				echo "   \"Adapter\": \"$adapter\","
				echo "   \"temp1\": \"$temp1\","
				echo "   \"high\": \"$high\","
				echo "   \"crit\": \"$crit\","
				echo "   \"hyst\": \"$hyst\""
				echo " }"
			fi
		fi
		bloque=""
		primer_bloque="0"
	else
		bloque+="$linea"$'\n'
	fi
done
echo "}"
