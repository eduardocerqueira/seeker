#date: 2025-01-28T16:56:53Z
#url: https://api.github.com/gists/bda65e81cb0fcd35a186d4c4c7556748
#owner: https://api.github.com/users/evtaccount

#!/usr/bin/env zsh

# --- ОТКЛЮЧАЕМ ВОЗМОЖНУЮ ОТЛАДКУ ---
set +x +v
unsetopt xtrace 2>/dev/null
unsetopt verbose 2>/dev/null

# --- ИМЯ VPN ПО УМОЛЧАНИЮ (ЕСЛИ НЕ ПЕРЕДАНО ВТОРЫМ АРГУМЕНТОМ) ---
DEFAULT_VPN_NAME="V2BOX"

# --- ФУНКЦИЯ ДЛЯ ИЗВЛЕЧЕНИЯ ИМЕН VPN-СЕРВИСОВ ---
function get_vpn_services() {
      # Получаем список VPN (строки, начинающиеся на "* (")
      local raw_services
      raw_services="$(scutil --nc list | grep '^\*')"

      if [[ -z "$raw_services" ]]; then
        echo "Не найдено ни одного VPN-сервиса в системных настройках."
        return 1
      fi

      # Составим массив строк
      local lines=()
      while IFS= read -r line; do
        lines+=("$line")
      done <<< "$raw_services"

      # Извлекаем имена сервисов (обычно в кавычках)
      local service_names=()
      for line in "${lines[@]}"; do
        name="$(sed -n 's/.*) "\(.*\)".*/\1/p' <<< "$line")"
        if [[ -z "$name" ]]; then
          name="$(sed -n 's/.*) //p' <<< "$line")"
        fi
        service_names+=("$name")
      done

      # Возвращаем список сервисов
      echo "${service_names[@]}"
}

# --- ФУНКЦИЯ ДЛЯ КОРОТКОГО СТАТУСА (ТОЛЬКО 1 СТРОКА) ---
function get_vpn_status() {
    scutil --nc status "$1" | head -n 1
}

# --- ПЕРЕКЛЮЧИТЬ (ПОДКЛЮЧИТЬ/ОТКЛЮЧИТЬ) VPN ---
function vpn_switch() {
    local service_names=($(get_vpn_services))
    
    # Определяем текущий подключенный VPN
    local current_vpn=""
    for srv in "${service_names[@]}"; do
        local st
        st="$(get_vpn_status "$srv")"
        if [[ "$st" == "Connected" ]]; then
            current_vpn="$srv"
            break
        fi
    done

    if [[ -z "$current_vpn" ]]; then
        # Если ни один VPN не подключён, подключаем VPN по умолчанию
        echo "VPN не подключен. Подключаем VPN по умолчанию: \"$DEFAULT_VPN_NAME\""
        scutil --nc start "$DEFAULT_VPN_NAME"
    else
        # Находим индекс текущего подключённого VPN
        local index=-1
        local count=${#service_names[@]}
        for (( i=0; i<=count; i++ )); do
            if [[ "${service_names[i]}" == "$current_vpn" ]]; then
                index="$i"
                break
            fi
        done

        if (( index == -1 )); then
            echo "Текущий VPN \"$current_vpn\" не найден в списке."
            return 1
        fi

        # Вычисляем следующий VPN в списке (циклический переход)
        local next_index=$(( (index + 1) ))
        if (( next_index > ${#service_names[@]} )); then
            next_index=1
        fi
        
        local next_vpn="${service_names[$next_index]}"

        echo "Отключаем текущий VPN: \"$current_vpn\""
        scutil --nc stop "$current_vpn"
        
        sleep 1

        echo "Подключаем следующий VPN: \"$next_vpn\""
        scutil --nc start "$next_vpn"
    fi
}

# --- ПОДКЛЮЧИТЬ ДЕФОЛТНЫЙ VPN, ЕСЛИ НИ ОДИН НЕ ПОДКЛЮЧЕН ---
function vpn_connect() {
    local service_names=($(get_vpn_services))

    # Проверяем список сервисов
    if [[ ${#service_names[@]} -eq 0 ]]; then
        echo "Список VPN пуст после фильтрации. Проверьте настройки."
        return 1
    fi

    # Проверяем, подключен ли какой-либо VPN
    local any_connected=0
    local connected=""
    for srv in "${service_names[@]}"; do
        local st
        st="$(get_vpn_status "$srv")"
        if [[ "$st" == "Connected" ]]; then
            any_connected=1
            connected=$srv
            break
        fi
    done

    if [[ $any_connected -eq 0 ]]; then
        # Если ни один VPN не подключён, подключаем VPN по умолчанию
        echo "VPN не подключен. Подключаем VPN по умолчанию: \"$DEFAULT_VPN_NAME\""
        scutil --nc start "$DEFAULT_VPN_NAME"
    else
        # Если VPN уже подключен, выводим сообщение
        echo "VPN уже подключен. Отключите текущий VPN, чтобы подключить другой."
    fi
}

# --- ФОРСИРОВАННО ОТКЛЮЧИТЬ VPN, ЕСЛИ ОН ПОДКЛЮЧЕН ---
function vpn_disconnect() {
    local service_names=($(get_vpn_services))

    local i=1
    for srv in "${service_names[@]}"; do
        st="$(get_vpn_status "$srv")"
        if [[ "$st" == "Connected" ]]; then
            echo "Отключаем VPN: \"$srv\""
            scutil --nc stop "$srv"
            break
        fi
        (( i++ ))
    done
}

# --- ПОКАЗАТЬ СТАТУС VPN ---
function vpn_status() {
    local service_names=($(get_vpn_services))

    echo "Статусы всех VPN-сервисов:"
    local i=1
    for srv in "${service_names[@]}"; do
        st="$(get_vpn_status "$srv")"
        echo "$i) $srv — $st"
        (( i++ ))
    done
}

# --- МЕНЮ: ПЕРЕКЛЮЧЕНИЕ ИЗ СПИСКА VPN ---
function vpn_menu() {
  local service_names=($(get_vpn_services))

  echo "Доступные VPN-сервисы:"
  local i=1
  for srv in "${service_names[@]}"; do
    st="$(get_vpn_status "$srv")"
    echo "$i) $srv — $st"
    (( i++ ))
  done

  echo
  echo -n "Выберите номер VPN для переключения (Enter — выход): "
  read user_choice

  if [[ -z "$user_choice" || ! "$user_choice" =~ ^[0-9]+$ ]]; then
    echo "Выход без переключения."
    return
  fi

  local count="${#service_names[@]}"
  if (( user_choice < 1 || user_choice > count )); then
    echo "Нет VPN с таким номером. Выход."
    return 1
  fi

  local chosen_service="${service_names[$((user_choice))]}"
  local cur_status
  cur_status="$(get_vpn_status "$chosen_service")"

  # Логика «switch» для выбранного VPN:
  case "$cur_status" in
    Connected)
      echo "Отключаем VPN: \"$chosen_service\""
      scutil --nc stop "$chosen_service"
      ;;
      
    Disconnected|*)
      i=1
      for srv in "${service_names[@]}"; do
        stt="$(get_vpn_status "$srv")"
        
        # Логика «switch» для выбранного VPN:
        if [[ "$stt" == "Connected" && srv != chosen_service ]]; then
          local disconnect_service="${service_names[$((i))]}"
          echo "Отключаем VPN: \"${service_names[$((i))]}\""
          scutil --nc stop "$disconnect_service"
        fi
        (( i++ ))
      done
      
      sleep 0.5
    
      echo "Подключаем VPN: \"$chosen_service\""
      scutil --nc start "$chosen_service"
      ;;
  esac
}

# --- ОБРАБОТКА АРГУМЕНТОВ ---
ACTION="$1"
VPN_NAME="$2"

# Если имя VPN не указано, берём DEFAULT_VPN_NAME
if [[ -z "$VPN_NAME" ]]; then
    VPN_NAME="$DEFAULT_VPN_NAME"
fi

case "$ACTION" in
    switch)
        vpn_switch
        ;;
    connect)
        vpn_connect
        ;;
    disconnect)
        vpn_disconnect
        ;;
    status)
        vpn_status
        ;;
    menu)
        vpn_menu
        ;;
    *)
        echo "Использование: $0 {switch|disconnect|status|menu} [VPN_NAME]"
        echo
        echo "  switch      - Переключить VPN: подключить следующий, отключить текущий и т.д."
        echo "  connect     - Подключить дефолтный VPN"
        echo "  disconnect  - Отключить все VPN"
        echo "  status      - Показать статусы всех VPN-сервисов"
        echo "  menu        - Показать список VPN и переключить выбранный пункт"
        exit 1
        ;;
esac
