#date: 2025-04-15T16:37:59Z
#url: https://api.github.com/gists/9dcc05cd92ddee9f28b7e73e729b497f
#owner: https://api.github.com/users/felipemagrassi

#!/bin/bash

BUCKET="s3tar"
REGION="us-east-1"
ROOT_PREFIX="raw/olx/olx_s"
PROFILE_ARG=""
HOJE=$(date +%s)

# Detecta sistema operacional
OS=$(uname)
USE_MACOS_DATE=false
[ "$OS" = "Darwin" ] && USE_MACOS_DATE=true

to_timestamp() {
  local date_str="$1"
  if $USE_MACOS_DATE; then
    date -j -f "%Y-%m-%d" "$date_str" "+%s" 2>/dev/null || echo 0
  else
    date -d "$date_str" "+%s" 2>/dev/null || echo 0
  fi
}

for arg in "$@"; do
  case $arg in
    --profile=*)
      PROFILE_NAME="${arg#*=}"
      PROFILE_ARG="--profile $PROFILE_NAME"
      ;;
  esac
done

echo "🔍 Iniciando coleta de caminhos para arquivamento..."
rm -f arquivos_para_processar.jsonl
touch arquivos_para_processar.jsonl

# Lista objetos principais
aws $PROFILE_ARG s3 ls "s3://$BUCKET/$ROOT_PREFIX/" | awk '/PRE/ {print $2}' | tr -d '/' | while read -r OBJECT; do
  BASE_PREFIX="$ROOT_PREFIX/$OBJECT"
  echo "📁 Explorando objeto: $OBJECT"

  aws $PROFILE_ARG s3 ls "s3://$BUCKET/$BASE_PREFIX/" | awk -F= '/year=/{print $2}' | tr -d '/' | while read -r YEAR; do
    echo "  📆 Ano: $YEAR"

    aws $PROFILE_ARG s3 ls "s3://$BUCKET/$BASE_PREFIX/year=$YEAR/" | awk -F= '/month=/{print $2}' | tr -d '/' | while read -r MONTH; do
      echo "    🗓️  Mês: $MONTH"

      aws $PROFILE_ARG s3 ls "s3://$BUCKET/$BASE_PREFIX/year=$YEAR/month=$MONTH/" | awk -F= '/day=/{print $2}' | tr -d '/' | while read -r DAY; do
        echo "      📅 Dia: $DAY"

        [ "$DAY" -eq 1 ] && echo "      ⚠️  Ignorando dia 1" && continue

        DATA_DIR=$(to_timestamp "$YEAR-$MONTH-$DAY")
        DIFF=$(( (HOJE - DATA_DIR) / 86400 ))

        if [ "$DATA_DIR" -eq 0 ]; then
          echo "      ❌ Data inválida: $YEAR-$MONTH-$DAY"
          continue
        fi

        if [ "$DIFF" -lt 90 ]; then
          echo "      ⏳ Ignorando $YEAR-$MONTH-$DAY (apenas $DIFF dias atrás)"
          continue
        fi

        FULL_PATH="$OBJECT/year=$YEAR/month=$MONTH/day=$DAY"
        echo "      ✅ Adicionando para processamento: $FULL_PATH"

        jq -n --arg object "$OBJECT" --arg year "$YEAR" --arg month "$MONTH" --arg day "$DAY" --arg path "$FULL_PATH" \
          '{object: $object, year: $year, month: $month, day: $day, path: $path}' >> arquivos_para_processar.jsonl
      done
    done
  done
done

echo "✅ JSON Lines gerado: arquivos_para_processar.jsonl"
echo "📦 Total de entradas: $(wc -l < arquivos_para_processar.jsonl)"

# Opcional: converter para JSON array tradicional
jq -s '.' arquivos_para_processar.jsonl > arquivos_para_processar.json
echo "🧾 JSON formatado salvo como: arquivos_para_processar.json"

