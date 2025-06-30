#date: 2025-06-30T16:54:00Z
#url: https://api.github.com/gists/7ffaef74ecf321f203467ff8cab6688a
#owner: https://api.github.com/users/sexol123

#!/bin/bash

LOG="/var/log/gpu-hotplug-daemon.log"
STATE_FILE="/tmp/gpu-last-state"
CONF_BAK="/etc/X11/xorg.conf.d/10-nvidia-only.bak"
CONF_ACTIVE="/etc/X11/xorg.conf.d/10-nvidia-only.conf"

export DISPLAY=:0
export XAUTHORITY=/home/sergei/.Xauthority

WANT_STATE=""

SLEEP_INTERVAL=5

# ----------------------------------------
# Почему два имени HDMI-порта?
#
# В Linux (особенно с разными драйверами NVIDIA/AMD, версиями ядра и X-сервера) имя HDMI-порта может динамически меняться.
# Например:
#  - Старое или первоначальное имя: "HDMI-1-0" (BEFOR_USE_HDMI_NAME)
#  - Текущее или новое имя: "HDMI-0" (USE_HDMI_NAME)
#
# Такое бывает из-за внутренней переинициализации драйверов или аппаратного переопределения.
# Поэтому, чтобы надёжно отследить подключение HDMI, скрипт проверяет оба варианта.
# Если любой из них подключён, считаем HDMI активным.
# ----------------------------------------
BEFOR_USE_HDMI_NAME="HDMI-1-0"
USE_HDMI_NAME="HDMI-0"

command -v kdialog >/dev/null || { echo "❌ kdialog не установлен"; exit 1; }
command -v xrandr >/dev/null || { echo "❌ xrandr не установлен"; exit 1; }

function log {
    echo "$(date '+%F %T') $1" >> "$LOG"
}

function get_connected_hdmi_port {
    if xrandr | grep -q "^$USE_HDMI_NAME connected"; then
        echo "$USE_HDMI_NAME"
        return 0
    elif xrandr | grep -q "^$BEFOR_USE_HDMI_NAME connected"; then
        echo "$BEFOR_USE_HDMI_NAME"
        return 0
    else
        echo ""
        return 1
    fi
}

function switch_to_nvidia {
    log "Пользователь подтвердил переключение на NVIDIA-only"
    echo "nvidia" > "$STATE_FILE"
    cp "$CONF_BAK" "$CONF_ACTIVE"
    sudo -u sergei kdialog --passivepopup "Переключение на NVIDIA (HDMI)" 5
    systemctl restart sddm
}

function switch_to_amd {
    log "HDMI отключён — переключаемся на AMD"
    echo "amd" > "$STATE_FILE"
    rm -f "$CONF_ACTIVE"
    sudo -u sergei kdialog --passivepopup "Переключение на встроенную графику (AMD)" 5
    systemctl restart sddm
}

while true; do
    if ! xrandr >/dev/null 2>&1; then
        log "⚠️ xrandr недоступен — возможно, X ещё не запущен"
        sleep $SLEEP_INTERVAL
        continue
    fi

    HDMI_PORT=$(get_connected_hdmi_port)

    if [ -n "$HDMI_PORT" ]; then
        HDMI_CONNECTED="yes"
    else
        HDMI_CONNECTED="no"
    fi

    STATE=$(cat "$STATE_FILE" 2>/dev/null)
    STATE=${STATE:-unknown}

    if [ "$HDMI_CONNECTED" == "yes" ]; then
        log "$HDMI_PORT подключён"

        if [ "$STATE" != "nvidia" ] && [ "$WANT_STATE" != "refused" ]; then
            log "$HDMI_PORT подключён — ожидаем подтверждения переключения на NVIDIA-only"

            if sudo -u sergei kdialog --yesno "$HDMI_PORT подключён. Переключиться на NVIDIA-only?" --title "Подтверждение переключения"; then
                switch_to_nvidia
            else
                log "Пользователь отказался — остаёмся на AMD (refused)"
                WANT_STATE="refused"
            fi
        fi

    else
        if [ "$STATE" != "amd" ]; then
            log "$HDMI_PORT отключён — переключаемся на AMD"
            switch_to_amd
            WANT_STATE=""
        fi
    fi

    sleep $SLEEP_INTERVAL
done
