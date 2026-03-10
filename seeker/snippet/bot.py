#date: 2026-03-10T17:35:48Z
#url: https://api.github.com/gists/7b5e64debc510c3ef9f71294b3c3c75a
#owner: https://api.github.com/users/77den77f

import requests
import time
import json
import os
import threading
from datetime import datetime

TOKEN = "8736588096: "**********"
CHAT_ID = "838037026"
SETTINGS_FILE = "screener_settings.json"
DAILY_FILE = "daily_signals.json"

DEFAULT_SETTINGS = {
    "price_enabled": True,
    "price_bybit": True,
    "price_binance": False,
    "oi_enabled": True,
    "price_pct": 10.0,
    "price_time": 15,
    "price_rollback": 3.0,
    "price_daily_limit": 5,
    "oi_pct": 10.0,
    "oi_time": 15,
    "oi_price_pct": 30.0,
    "oi_daily_limit": 5,
    "direction": "both",
    "delay": 0
}

# Состояния меню
user_state = {}
waiting_edit = {}

PARAM_LABELS = {
    "price_pct": "% движения цены",
    "price_time": "Время цены (мин)",
    "price_rollback": "Откат %",
    "price_daily_limit": "Лимит цены/день",
    "oi_pct": "% движения OI",
    "oi_time": "Время OI (мин)",
    "oi_price_pct": "% цены при OI",
    "oi_daily_limit": "Лимит OI/день",
}

def load_settings():
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE) as f:
            s = json.load(f)
        for k, v in DEFAULT_SETTINGS.items():
            if k not in s:
                s[k] = v
        return s
    return DEFAULT_SETTINGS.copy()

def save_settings(s):
    with open(SETTINGS_FILE, "w") as f:
        json.dump(s, f)

def load_daily():
    today = datetime.utcnow().strftime("%Y-%m-%d")
    if os.path.exists(DAILY_FILE):
        with open(DAILY_FILE) as f:
            data = json.load(f)
        if data.get("date") == today:
            return data
    return {"date": today, "counts": {}}

def save_daily(d):
    with open(DAILY_FILE, "w") as f:
        json.dump(d, f)

def inc_daily(key, daily):
    today = datetime.utcnow().strftime("%Y-%m-%d")
    if daily.get("date") != today:
        daily = {"date": today, "counts": {}}
    daily["counts"][key] = daily["counts"].get(key, 0) + 1
    save_daily(daily)
    return daily

def get_daily_count(key, daily):
    today = datetime.utcnow().strftime("%Y-%m-%d")
    if daily.get("date") != today:
        return 0
    return daily["counts"].get(key, 0)

def send_msg(text, reply_markup=None):
    url = "https: "**********"
    data = {"chat_id": CHAT_ID, "text": text}
    if reply_markup:
        data["reply_markup"] = json.dumps(reply_markup)
    try:
        requests.post(url, data=data, timeout=10)
    except:
        pass

def answer_cb(cbid):
    try:
        requests.post("https: "**********"
                      data={"callback_query_id": cbid}, timeout=5)
    except:
        pass

def set_commands():
    cmds = [
        {"command": "start", "description": "Запустить"},
        {"command": "test", "description": "Проверка"}
    ]
    try:
        requests.post("https: "**********"
                      json={"commands": cmds}, timeout=10)
    except:
        pass

def yn(v):
    return "✅" if v else "❌"

def dir_str(d):
    return {"both": "Памп+Дамп", "pump": "Только Памп", "dump": "Только Дамп"}[d]

# ==================== КЛАВИАТУРЫ ====================

def kb_main():
    return {
        "keyboard": [
            [{"text": "📊 Скринер"}, {"text": "✅ Тест"}]
        ],
        "resize_keyboard": True,
        "persistent": True
    }

def kb_settings(s):
    return {
        "keyboard": [
            [{"text": "💰 Цена " + yn(s["price_enabled"])}, {"text": "📈 OI " + yn(s["oi_enabled"])}],
            [{"text": "🔙 Назад"}]
        ],
        "resize_keyboard": True,
        "persistent": True
    }

def kb_price(s):
    return {
        "keyboard": [
            [{"text": "📈 Движение: " + str(s["price_pct"]) + "%"}],
            [{"text": "⏱ Время: " + str(s["price_time"]) + " мин"}],
            [{"text": "↩️ Откат: " + str(s["price_rollback"]) + "%"}],
            [{"text": "📋 Лимит/день: " + str(s["price_daily_limit"])}],
            [{"text": "🔀 Направление: " + dir_str(s["direction"])}],
            [{"text": "Bybit " + yn(s["price_bybit"])}, {"text": "Binance " + yn(s["price_binance"])}],
            [{"text": "Скринер цены " + ("✅ Вкл" if s["price_enabled"] else "❌ Выкл")}],
            [{"text": "🔙 Назад"}]
        ],
        "resize_keyboard": True,
        "persistent": True
    }

def kb_oi(s):
    return {
        "keyboard": [
            [{"text": "📊 % OI: " + str(s["oi_pct"]) + "%"}],
            [{"text": "⏱ Время OI: " + str(s["oi_time"]) + " мин"}],
            [{"text": "💰 % цены при OI: " + str(s["oi_price_pct"]) + "%"}],
            [{"text": "📋 Лимит OI/день: " + str(s["oi_daily_limit"])}],
            [{"text": "Скринер OI " + ("✅ Вкл" if s["oi_enabled"] else "❌ Выкл")}],
            [{"text": "🔙 Назад"}]
        ],
        "resize_keyboard": True,
        "persistent": True
    }

def settings_text(s):
    return (
        "📊 <b>Крипто Скринер TreiDen</b>\n\n"
        "<b>💰 Цена</b> " + yn(s["price_enabled"]) + "\n"
        "Bybit: " + yn(s["price_bybit"]) + " | Binance: " + yn(s["price_binance"]) + "\n"
        "Направление: " + dir_str(s["direction"]) + "\n"
        "Движение: <code>" + str(s["price_pct"]) + "%</code> за <code>" + str(s["price_time"]) + " мин</code>\n"
        "Откат: <code>" + str(s["price_rollback"]) + "%</code>\n"
        "Лимит/день: <code>" + str(s["price_daily_limit"]) + "</code>\n\n"
        "<b>📈 OI</b> " + yn(s["oi_enabled"]) + "\n"
        "Движение OI: <code>" + str(s["oi_pct"]) + "%</code> за <code>" + str(s["oi_time"]) + " мин</code>\n"
        "% цены при OI: <code>" + str(s["oi_price_pct"]) + "%</code>\n"
        "Лимит OI/день: <code>" + str(s["oi_daily_limit"]) + "</code>"
    )

# ==================== BYBIT/BINANCE API ====================

def get_bybit_tickers():
    try:
        r = requests.get("https://api.bybit.com/v5/market/tickers?category=linear", timeout=5).json()
        result = {}
        for item in r["result"]["list"]:
            if item["symbol"].endswith("USDT"):
                result[item["symbol"]] = {
                    "price": float(item["lastPrice"]),
                    "volume": float(item["volume24h"]),
                    "funding": float(item.get("fundingRate", 0))
                }
        return result
    except:
        return {}

def get_binance_tickers():
    try:
        r = requests.get("https://fapi.binance.com/fapi/v1/ticker/price", timeout=5).json()
        result = {}
        for item in r:
            if item["symbol"].endswith("USDT"):
                result[item["symbol"]] = {"price": float(item["price"]), "volume": 0, "funding": 0}
        return result
    except:
        return {}

def get_bybit_oi(symbol):
    try:
        r = requests.get(
            "https://api.bybit.com/v5/market/open-interest?category=linear&symbol=" + symbol +
            "&intervalTime=5min&limit=1", timeout=5
        ).json()
        return float(r["result"]["list"][0]["openInterest"])
    except:
        return None

def get_binance_oi(symbol):
    try:
        r = requests.get("https://fapi.binance.com/fapi/v1/openInterest?symbol=" + symbol, timeout=5).json()
        return float(r["openInterest"])
    except:
        return None

def get_bybit_ls(symbol):
    try:
        r = requests.get(
            "https://api.bybit.com/v5/market/account-ratio?category=linear&symbol=" + symbol +
            "&period=1h&limit=1", timeout=5
        ).json()
        val = float(r["result"]["list"][0]["buyRatio"])
        return round(val / (1 - val), 2) if val < 1 else round(val, 2)
    except:
        return None

def get_binance_ls(symbol):
    try:
        r = requests.get(
            "https://fapi.binance.com/futures/data/globalLongShortAccountRatio?symbol=" + symbol +
            "&period=1h&limit=1", timeout=5
        ).json()
        return float(r[0]["longShortRatio"])
    except:
        return None

def get_bybit_liquidations(symbol):
    try:
        r = requests.get(
            "https://api.bybit.com/v5/market/liquidation?category=linear&symbol=" + symbol +
            "&limit=50", timeout=5
        ).json()
        longs = 0.0
        shorts = 0.0
        for item in r["result"]["list"]:
            val = float(item["size"]) * float(item["price"])
            if item["side"] == "Buy":
                shorts += val
            else:
                longs += val
        return longs, shorts
    except:
        return None, None

def fmt_dolf(symbol, ex, oi_pct, funding, ls, liq_long, liq_short):
    lines = ["—————", "📊 DOLF:"]
    if oi_pct is not None:
        icon = "✅" if oi_pct < -1.0 else "🟡" if oi_pct < 0 else "🔴"
        lines.append("OI: " + ("+" if oi_pct > 0 else "") + str(round(oi_pct, 2)) + "% " + icon)
    if funding is not None:
        icon = "✅" if -0.05 <= funding * 100 <= 0.05 else "🔴"
        lines.append("Funding: " + ("+" if funding > 0 else "") + str(round(funding * 100, 4)) + "% " + icon)
    if ls is not None:
        icon = "✅" if ls >= 1.2 else "🟡" if ls >= 1.0 else "🔴"
        lines.append("L/S: " + str(ls) + " " + icon)
    if liq_long is not None and liq_short is not None and (liq_long > 0 or liq_short > 0):
        lines.append("Ликв: лонги $" + str(int(liq_long)) + " / шорты $" + str(int(liq_short)))
    return "\n".join(lines)

# ==================== ОБРАБОТКА СООБЩЕНИЙ ====================

def get_updates(offset=None):
    try:
        r = requests.get(
            "https: "**********"
            params={"timeout": 5, "offset": offset}, timeout=10
        ).json()
        return r.get("result", [])
    except:
        return []

def handle_message(uid, msg, s):
    # Ожидаем ввод значения
    if uid in waiting_edit:
        param = waiting_edit.pop(uid)
        try:
            val = float(msg)
            s[param] = val
            save_settings(s)
            label = PARAM_LABELS.get(param, param)
            state = user_state.get(uid, "main")
            if state == "price":
                send_msg("✅ " + label + " = <code>" + str(val) + "</code>",
                         reply_markup=kb_price(s))
            elif state == "oi":
                send_msg("✅ " + label + " = <code>" + str(val) + "</code>",
                         reply_markup=kb_oi(s))
            else:
                send_msg("✅ " + label + " = <code>" + str(val) + "</code>")
        except:
            send_msg("❌ Введи число")
        return

    if msg in ["/start", "/help"]:
        user_state[uid] = "main"
        send_msg("🚀 Крипто Скринер TreiDen!", reply_markup=kb_main())
        return

    if msg in ["✅ Тест", "/test"]:
        send_msg("✅ Скринер работает!\n🕐 " + datetime.utcnow().strftime("%H:%M") + " UTC")
        return

    if msg == "📊 Скринер":
        user_state[uid] = "settings"
        send_msg(settings_text(s), reply_markup=kb_settings(s))
        return

    if msg == "🔙 Назад":
        state = user_state.get(uid, "main")
        if state in ["price", "oi"]:
            user_state[uid] = "settings"
            send_msg(settings_text(s), reply_markup=kb_settings(s))
        else:
            user_state[uid] = "main"
            send_msg("Главное меню", reply_markup=kb_main())
        return

    if msg.startswith("💰 Цена"):
        user_state[uid] = "price"
        send_msg("Настройки цены:", reply_markup=kb_price(s))
        return

    if msg.startswith("📈 OI"):
        user_state[uid] = "oi"
        send_msg("Настройки OI:", reply_markup=kb_oi(s))
        return

    # Переключатели в меню цены
    if msg.startswith("Bybit"):
        s["price_bybit"] = not s["price_bybit"]
        save_settings(s)
        send_msg("✅ Bybit " + ("включён" if s["price_bybit"] else "выключен"),
                 reply_markup=kb_price(s))
        return

    if msg.startswith("Binance"):
        s["price_binance"] = not s["price_binance"]
        save_settings(s)
        send_msg("✅ Binance " + ("включён" if s["price_binance"] else "выключен"),
                 reply_markup=kb_price(s))
        return

    if msg.startswith("Скринер цены"):
        s["price_enabled"] = not s["price_enabled"]
        save_settings(s)
        send_msg("✅ Скринер цены " + ("включён" if s["price_enabled"] else "выключен"),
                 reply_markup=kb_price(s))
        return

    if msg.startswith("Скринер OI"):
        s["oi_enabled"] = not s["oi_enabled"]
        save_settings(s)
        send_msg("✅ Скринер OI " + ("включён" if s["oi_enabled"] else "выключен"),
                 reply_markup=kb_oi(s))
        return

    if msg.startswith("🔀 Направление"):
        order = ["both", "pump", "dump"]
        s["direction"] = order[(order.index(s["direction"]) + 1) % 3]
        save_settings(s)
        send_msg("✅ Направление: " + dir_str(s["direction"]), reply_markup=kb_price(s))
        return

    # Параметры цены — нажали на кнопку
    price_map = {
        "📈 Движение:": "price_pct",
        "⏱ Время:": "price_time",
        "↩️ Откат:": "price_rollback",
        "📋 Лимит/день:": "price_daily_limit",
    }
    if user_state.get(uid) == "price":
        for prefix, param in price_map.items():
            if msg.startswith(prefix):
                waiting_edit[uid] = param
                label = PARAM_LABELS.get(param, param)
                send_msg("✏️ Введи значение для <b>" + label + "</b>:\nТекущее: <code>" +
                         str(s[param]) + "</code>")
                return

    # Параметры OI — нажали на кнопку
    oi_map = {
        "📊 % OI:": "oi_pct",
        "⏱ Время OI:": "oi_time",
        "💰 % цены при OI:": "oi_price_pct",
        "📋 Лимит OI/день:": "oi_daily_limit",
    }
    if user_state.get(uid) == "oi":
        for prefix, param in oi_map.items():
            if msg.startswith(prefix):
                waiting_edit[uid] = param
                label = PARAM_LABELS.get(param, param)
                send_msg("✏️ Введи значение для <b>" + label + "</b>:\nТекущее: <code>" +
                         str(s[param]) + "</code>")
                return

def cmd_thread():
    offset = None
    updates = get_updates()
    if updates:
        offset = updates[-1]["update_id"] + 1

    while True:
        try:
            updates = get_updates(offset)
            for u in updates:
                offset = u["update_id"] + 1
                s = load_settings()

                if "message" in u:
                    msg = u["message"].get("text", "")
                    uid = str(u["message"]["from"]["id"])
                    handle_message(uid, msg, s)
        except:
            pass
        time.sleep(1)

def screener():
    ph = {}
    oih = {}
    last_signal = {}
    daily = load_daily()
    oi_cache = {}
    while True:
        try:
            s = load_settings()
            daily = load_daily()
            now = time.time()
            pw = s["price_time"] * 60
            ow = s["oi_time"] * 60
            all_data = {}
            if s.get("price_bybit", True):
                for sym, d in get_bybit_tickers().items():
                    all_data[sym + "_BB"] = {"ex": "Bybit", "sym": sym, "price": d["price"], "vol": d["volume"], "funding": d["funding"]}
            if s.get("price_binance", False):
                for sym, d in get_binance_tickers().items():
                    all_data[sym + "_BN"] = {"ex": "Binance", "sym": sym, "price": d["price"], "vol": d["volume"], "funding": d["funding"]}
            for key, d in all_data.items():
                sym = d["sym"]
                ex = d["ex"]
                price = d["price"]
                vol = d["vol"]
                funding = d["funding"]
                if key not in ph:
                    ph[key] = []
                ph[key].append((now, price))
                ph[key] = [(t, p) for t, p in ph[key] if now - t <= pw + 120]
                if key not in oi_cache:
                    oi_cache[key] = []
                oi_now = get_bybit_oi(sym) if ex == "Bybit" else get_binance_oi(sym)
                if oi_now:
                    oi_cache[key].append((now, oi_now))
                    oi_cache[key] = [(t, v) for t, v in oi_cache[key] if now - t <= pw + 120]
                if s["price_enabled"] and len(ph[key]) >= 2:
                    old = [(t, p) for t, p in ph[key] if now - t >= 120]
                    if old:
                        old_price = old[0][1]
                        pct = ((price - old_price) / old_price) * 100
                        prices = [p for t, p in ph[key]]
                        if pct > 0:
                            rb = ((max(prices) - price) / max(prices)) * 100
                        else:
                            mn = min(prices)
                            rb = ((price - mn) / mn) * 100 if mn > 0 else 0
                        is_pump = pct >= s["price_pct"]
                        is_dump = pct <= -s["price_pct"]
                        rb_ok = abs(rb) <= s["price_rollback"]
                        dir_ok = s["direction"] == "both" or (s["direction"] == "pump" and is_pump) or (s["direction"] == "dump" and is_dump)
                        dlim_ok = get_daily_count(key + "_p", daily) < s["price_daily_limit"]
                        delay_ok = now - last_signal.get(key + "_p", 0) >= s.get("delay", 0)
                        if (is_pump or is_dump) and rb_ok and dir_ok and dlim_ok and delay_ok:
                            if is_pump:
                                icon = "🟢⬆ PUMP"
                            else:
                                icon = "🔴⬇ DUMP"
                            msg = (icon + ": " + sym + " (" + ex + ")\n" +
                                   "Цена: " + str(round(old_price, 6)) + " -> " + str(round(price, 6)) +
                                   " (" + ("+" if pct > 0 else "") + str(round(pct, 2)) + "%)\n")
                            if vol > 0:
                                msg += "Объём: " + str(int(vol)) + "\n"
                            msg += ("Сигналов/день: " + str(get_daily_count(key + "_p", daily) + 1) + "\n" +
                                    "Время: " + datetime.utcnow().strftime("%H:%M"))
                            oi_pct_val = None
                            if len(oi_cache[key]) >= 2:
                                old_oi_e = [(t, v) for t, v in oi_cache[key] if now - t >= 120]
                                if old_oi_e and oi_now:
                                    oi_pct_val = ((oi_now - old_oi_e[0][1]) / old_oi_e[0][1]) * 100
                            ls = get_bybit_ls(sym) if ex == "Bybit" else get_binance_ls(sym)
                            liq_long, liq_short = get_bybit_liquidations(sym) if ex == "Bybit" else (None, None)
                            msg += "\n" + fmt_dolf(sym, ex, oi_pct_val, funding, ls, liq_long, liq_short)
                            send_msg(msg)
                            daily = inc_daily(key + "_p", daily)
                            last_signal[key + "_p"] = now
                            ph[key] = [(now, price)]
                if s["oi_enabled"] and len(oi_cache[key]) >= 2:
                    old_oi_e = [(t, v) for t, v in oi_cache[key] if now - t >= 120]
                    if old_oi_e and oi_now:
                        old_oi = old_oi_e[0][1]
                        oi_pct = ((oi_now - old_oi) / old_oi) * 100
                        pe = [(t, p) for t, p in ph.get(key, []) if now - t >= 120]
                        p_start = pe[0][1] if pe else price
                        p_chg = ((price - p_start) / p_start) * 100
                        oi_dlim = get_daily_count(key + "_oi", daily) < s["oi_daily_limit"]
                        oi_delay = now - last_signal.get(key + "_oi", 0) >= s.get("delay", 0)
                        is_oi_p = oi_pct >= s["oi_pct"]
                        is_oi_d = oi_pct <= -s["oi_pct"]
                        if (is_oi_p or is_oi_d) and abs(p_chg) <= s["oi_price_pct"] and oi_dlim and oi_delay:
                            if is_oi_p:
                                icon = "🟢⬆ OI PUMP"
                            else:
                                icon = "🔴⬇ OI DUMP"
                            ls = get_bybit_ls(sym) if ex == "Bybit" else get_binance_ls(sym)
                            liq_long, liq_short = get_bybit_liquidations(sym) if ex == "Bybit" else (None, None)
                            msg = (icon + ": " + sym + " (" + ex + ")\n" +
                                   "OI: " + str(round(old_oi, 0)) + " -> " + str(round(oi_now, 0)) +
                                   " (" + ("+" if oi_pct > 0 else "") + str(round(oi_pct, 2)) + "%)\n" +
                                   "Цена: " + str(round(p_start, 6)) + " -> " + str(round(price, 6)) +
                                   " (" + ("+" if p_chg > 0 else "") + str(round(p_chg, 2)) + "%)\n" +
                                   "Сигналов/день: " + str(get_daily_count(key + "_oi", daily) + 1) + "\n" +
                                   "Время: " + datetime.utcnow().strftime("%H:%M") + "\n" +
                                   fmt_dolf(sym, ex, oi_pct, funding, ls, liq_long, liq_short))
                            send_msg(msg)
                            daily = inc_daily(key + "_oi", daily)
                            last_signal[key + "_oi"] = now
                time.sleep(0.05)
        except Exception:
            pass
        time.sleep(60)

def main():
    set_commands()
    t = threading.Thread(target=cmd_thread, daemon=True)
    t.start()
    send_msg("🚀 Крипто Скринер TreiDen запущен!\n/test", reply_markup=kb_main())
    screener()

if __name__ == "__main__":
    main()ue)
    t.start()
    send_msg("🚀 Крипто Скринер TreiDen запущен!\n/test", reply_markup=kb_main())
    screener()

if __name__ == "__main__":
    main()