#date: 2025-04-24T16:47:35Z
#url: https://api.github.com/gists/69f71d99ab30e8825be6089fc742d08b
#owner: https://api.github.com/users/jazzl0ver

#!/bin/bash

# OpenSearch endpoint
HOST="https://localhost:9200"

username=admin
read -s -p "Password: "**********"
echo

# Опционально: если нужна basic auth
AUTH="-u $username: "**********"

# Функция для обработки ошибок
handle_error() {
  echo "❌ Ошибка: $1"
  exit 1
}

# Получаем список всех индексов с primary shards > 1
indices=$(curl -ks $AUTH "$HOST/_cat/indices?h=index,pri" | awk '$2 > 1 {print $1}')
if [ -z "$indices" ]; then
  handle_error "Нет индексов с количеством primary shards > 1."
fi

for index in $indices; do
  tmp_index="${index}-tmp"

  echo "▶ Обрабатывается индекс: $index"

  # Получаем настройки и маппинг исходного индекса
  settings_file=$(mktemp)
  mappings_file=$(mktemp)
  all_settings_file=$(mktemp)

  curl -ks $AUTH "$HOST/$index" | jq -r ".\"$index\".settings.index" | jq 'del(."creation_date",."uuid",."version",."provided_name",."routing")' > "$settings_file"
  if [ $? -ne 0 ]; then handle_error "Ошибка получения настроек индекса $index"; fi

  curl -ks $AUTH "$HOST/$index/_mapping" | jq -r ".\"$index\".mappings" > "$mappings_file"
  if [ $? -ne 0 ]; then handle_error "Ошибка получения маппинга индекса $index"; fi

  # Получаем количество документов в исходном индексе
  doc_count_source=$(curl -ks $AUTH "$HOST/$index/_count" | jq '.count')
  if [ $? -ne 0 ]; then handle_error "Ошибка получения количества документов для индекса $index"; fi

  echo "📦 Создание временного индекса: $tmp_index с 1 primary шардом"

  # Создание временного индекса с 1 primary и теми же settings/mappings
  echo "
  {
    \"settings\": $(<"$settings_file" jq '. + {number_of_shards: 1, number_of_replicas: 1}'),
    \"mappings\": $(<"$mappings_file")
  }" > $all_settings_file
  curl -ks -XPUT $AUTH "$HOST/$tmp_index" -H 'Content-Type: application/json' -d @$all_settings_file > /dev/null
  if [ $? -ne 0 ]; then handle_error "Ошибка создания временного индекса $tmp_index"; fi

sleep 2

  echo "🔄 Reindex $index → $tmp_index"
  curl -ks -XPOST $AUTH "$HOST/_reindex?wait_for_completion=true" -H 'Content-Type: application/json' -d "
  {
    \"source\": { \"index\": \"$index\" },
    \"dest\": { \"index\": \"$tmp_index\" }
  }" > /dev/null
  if [ $? -ne 0 ]; then handle_error "Ошибка выполнения reindex для $index"; fi

  echo "⏳ Ожидание завершения реиндексации..."

  # Проверка результата реиндексации
  c=0
  while true; do
    # Получаем количество документов в временном индексе
    doc_count_tmp=$(curl -ks $AUTH "$HOST/$tmp_index/_count" | jq '.count')
    if [ $? -ne 0 ]; then handle_error "Ошибка получения количества документов для временного индекса $tmp_index"; fi

    # Проверка совпадения количества документов
    if [ "$doc_count_source" -eq "$doc_count_tmp" ]; then
      break
    fi

#    if [ "$c" -gt 6 ]; then
#      handle_error "Количество документов в исходном индексе ($doc_count_new) и временном индексе ($doc_count_tmp) не совпадает после $(($c*5)) секунд."
#    fi
    sleep 5  # Ждем 5 секунд перед повторной проверкой
    c=$(($c+1))
  done

  echo "🗑 Удаление исходного индекса: $index"
  curl -ks -XDELETE $AUTH "$HOST/$index" > /dev/null
  if [ $? -ne 0 ]; then handle_error "Ошибка удаления индекса $index"; fi

  # Создание основного индекса с 1 primary и теми же settings/mappings
  echo "📦 Создание основного индекса $index с 1 primary шардом"
  echo "
  {
    \"settings\": $(<"$settings_file" jq '. + {number_of_shards: 1, number_of_replicas: 1}'),
    \"mappings\": $(<"$mappings_file")
  }" > $all_settings_file
  curl -ks -XPUT $AUTH "$HOST/$index" -H 'Content-Type: application/json' -d @$all_settings_file > /dev/null
  if [ $? -ne 0 ]; then handle_error "Ошибка создания основного индекса $index"; fi

sleep 2
  echo "🔁 Reindex обратно: $tmp_index → $index"
  curl -ks -XPOST $AUTH "$HOST/_reindex?wait_for_completion=true" -H 'Content-Type: application/json' -d "
  {
    \"source\": { \"index\": \"$tmp_index\" },
    \"dest\": { \"index\": \"$index\" }
  }" > /dev/null
  if [ $? -ne 0 ]; then handle_error "Ошибка выполнения reindex обратно для $tmp_index → $index"; fi

  echo "⏳ Ожидание завершения реиндексации..."

  # Проверка результата реиндексации
  c=0
  while true; do
    # Получаем количество документов в основном индексе
    doc_count_new=$(curl -ks $AUTH "$HOST/$index/_count" | jq '.count')
    if [ $? -ne 0 ]; then handle_error "Ошибка получения количества документов для индекса $index"; fi

    # Проверка совпадения количества документов
    if [ "$doc_count_new" -eq "$doc_count_tmp" ]; then
      break
    fi

#    if [ "$c" -gt 10 ]; then
#      handle_error "Количество документов в основном индексе ($doc_count_new) и временном индексе ($doc_count_tmp) не совпадает после $(($c*5)) секунд."
#    fi
    sleep 5  # Ждем 5 секунд перед повторной проверкой
    c=$(($c+1))
  done

  echo "🧹 Удаление временного индекса: $tmp_index"
  curl -ks -XDELETE $AUTH "$HOST/$tmp_index" > /dev/null
  if [ $? -ne 0 ]; then handle_error "Ошибка удаления временного индекса $tmp_index"; fi

  # Удаляем временные файлы
  rm "$settings_file" "$mappings_file" "$all_settings_file"

  echo "✅ Готово: $index перешарден"
  echo ""
done
