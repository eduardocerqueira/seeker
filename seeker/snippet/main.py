#date: 2025-09-03T16:50:00Z
#url: https://api.github.com/gists/4e4d45502d8dd95c8c2b6654513d57a1
#owner: https://api.github.com/users/Jan3u

import os
import json
import logging
import time
from flask import Flask, request, abort
import telebot
from telebot import types

try:
    import requests
except Exception:
    requests = None

# ----------------- Config (env) -----------------
BOT_TOKEN = "**********"
ADMIN_IDS_ENV = os.getenv('ADMIN_IDS') or os.getenv('ADMIN_CHAT_ID')
if ADMIN_IDS_ENV:
    ADMIN_IDS = set(int(x.strip()) for x in ADMIN_IDS_ENV.split(',') if x.strip())
else:
    ADMIN_IDS = set()

CHANNEL_USERNAME = os.getenv('CHANNEL_USERNAME')
CHANNEL_ID = os.getenv('CHANNEL_ID')
WEBHOOK_BASE = os.getenv('WEBHOOK_BASE')
WEBHOOK_PATH = '/webhook'
WEBHOOK_URL = (WEBHOOK_BASE + WEBHOOK_PATH) if WEBHOOK_BASE else None
TELEGRAPH_PAY_LINK = os.getenv('TELEGRAPH_PAY_LINK', '')
CONTENT_FILE = os.getenv('CONTENT_FILE', 'content.json')

GIST_TOKEN = "**********"
GIST_ID = os.getenv('GIST_ID')        # optional
GIST_FILENAME_BANS = 'bans.json'
GIST_FILENAME_PENDING = 'pending.json'

# storage paths (local fallback)
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
BANS_FILE_LOCAL = os.path.join(DATA_DIR, 'bans.json')
PENDING_FILE_LOCAL = os.path.join(DATA_DIR, 'pending.json')

 "**********"i "**********"f "**********"  "**********"n "**********"o "**********"t "**********"  "**********"B "**********"O "**********"T "**********"_ "**********"T "**********"O "**********"K "**********"E "**********"N "**********": "**********"
    raise RuntimeError('BOT_TOKEN is required')

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

bot = "**********"=False)
app = Flask(__name__)

# ----------------- Rate limiting -----------------
user_rate = {}
RATE_LIMIT_WINDOW = 60
RATE_LIMIT_MAX = 40
RATE_MIN_INTERVAL = 0.25

def check_rate(user_id: int):
    now = time.time()
    s = user_rate.get(user_id)
    if not s:
        user_rate[user_id] = {'count': 1, 'window_start': now, 'last_ts': now, 'blocked_until': 0}
        return True, None
    if now < s.get('blocked_until', 0):
        return False, '😵There are too many requests. Try again later.🚦'
    if now - s.get('last_ts', 0) < RATE_MIN_INTERVAL:
        s['blocked_until'] = now + 3
        user_rate[user_id] = s
        return False, 'Too many quick actions. Wait a second.🕘'
    if now - s.get('window_start', 0) > RATE_LIMIT_WINDOW:
        s['window_start'] = now
        s['count'] = 1
        s['last_ts'] = now
        user_rate[user_id] = s
        return True, None
    s['count'] = s.get('count', 0) + 1
    s['last_ts'] = now
    if s['count'] > RATE_LIMIT_MAX:
        s['blocked_until'] = now + 30
        user_rate[user_id] = s
        return False, 'The request limit has been exceeded. Try again in 30 seconds.⏳'
    user_rate[user_id] = s
    return True, None

# ----------------- In-memory state -----------------
user_state = {}       # survey states
pending = {}          # admin_id -> target_user_id (persisted)
banned_users = set()  # persisted
recent_messages = []  # for /admin preview
RECENT_MAX = 50

# ----------------- Persistence (Gist/local) -----------------
GITHUB_API = 'https://api.github.com'

def gist_headers():
    return {'Authorization': "**********"

def create_gist_with_files():
    if not requests:
        logger.error('requests required to create gist')
        return None
    payload = {
        'description': 'Telegram bot storage for bans and pending',
        'public': False,
        'files': {
            GIST_FILENAME_BANS: {'content': '[]'},
            GIST_FILENAME_PENDING: {'content': '{}'}
        }
    }
    try:
        r = requests.post(f'{GITHUB_API}/gists', headers=gist_headers(), json=payload, timeout=20)
        if r.status_code in (200, 201):
            data = r.json()
            gid = data.get('id')
            logger.info('Created gist %s', gid)
            return gid
        logger.error('Failed to create gist: %s %s', r.status_code, r.text)
        return None
    except Exception:
        logger.exception('create_gist_with_files failed')
        return None

def load_gist_files(gist_id):
    if not requests:
        return None
    try:
        r = requests.get(f'{GITHUB_API}/gists/{gist_id}', headers=gist_headers(), timeout=20)
        if r.status_code != 200:
            logger.error('Failed to load gist %s: %s %s', gist_id, r.status_code, r.text)
            return None
        data = r.json()
        files = data.get('files', {})
        bans_content = files.get(GIST_FILENAME_BANS, {}).get('content')
        pending_content = files.get(GIST_FILENAME_PENDING, {}).get('content')
        try:
            bans = json.loads(bans_content) if bans_content else []
        except Exception:
            bans = []
        try:
            pend = json.loads(pending_content) if pending_content else {}
        except Exception:
            pend = {}
        return bans, pend
    except Exception:
        logger.exception('load_gist_files failed')
        return None

def update_gist_files(gist_id, bans_list, pending_dict):
    if not requests:
        return False
    payload = {'files': {
        GIST_FILENAME_BANS: {'content': json.dumps(bans_list, ensure_ascii=False, indent=2)},
        GIST_FILENAME_PENDING: {'content': json.dumps(pending_dict, ensure_ascii=False, indent=2)}
    }}
    try:
        r = requests.patch(f'{GITHUB_API}/gists/{gist_id}', headers=gist_headers(), json=payload, timeout=20)
        if r.status_code in (200, 201):
            return True
        logger.error('Failed to update gist %s: %s %s', gist_id, r.status_code, r.text)
        return False
    except Exception:
        logger.exception('update_gist_files failed')
        return False

def load_persisted_state():
    global pending, banned_users, GIST_ID
 "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"G "**********"I "**********"S "**********"T "**********"_ "**********"T "**********"O "**********"K "**********"E "**********"N "**********": "**********"
        if not GIST_ID:
            gid = create_gist_with_files()
            if gid:
                GIST_ID = gid
                logger.info('Set GIST_ID=%s (add to env for future restarts)', GIST_ID)
        if GIST_ID and requests:
            res = load_gist_files(GIST_ID)
            if res:
                bans_list, pend = res
                banned_users.clear(); banned_users.update(int(x) for x in bans_list)
                pending.clear(); pending.update({int(k): int(v) for k, v in pend.items()})
                logger.info('Loaded persisted state from gist %s', GIST_ID)
                return
            else:
                logger.warning('Could not load gist content, falling back to local files')
    # fallback local
    try:
        if os.path.exists(BANS_FILE_LOCAL):
            with open(BANS_FILE_LOCAL, 'r', encoding='utf-8') as f:
                bl = json.load(f)
                banned_users.clear(); banned_users.update(int(x) for x in bl)
        if os.path.exists(PENDING_FILE_LOCAL):
            with open(PENDING_FILE_LOCAL, 'r', encoding='utf-8') as f:
                pd = json.load(f)
                pending.clear(); pending.update({int(k): int(v) for k, v in pd.items()})
        logger.info('Loaded persisted state from local files')
    except Exception:
        logger.exception('Failed to load local persisted state')

def persist_state():
    try:
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"G "**********"I "**********"S "**********"T "**********"_ "**********"T "**********"O "**********"K "**********"E "**********"N "**********"  "**********"a "**********"n "**********"d "**********"  "**********"G "**********"I "**********"S "**********"T "**********"_ "**********"I "**********"D "**********"  "**********"a "**********"n "**********"d "**********"  "**********"r "**********"e "**********"q "**********"u "**********"e "**********"s "**********"t "**********"s "**********": "**********"
            success = update_gist_files(GIST_ID, list(banned_users), {str(k): v for k, v in pending.items()})
            if success:
                return
        with open(BANS_FILE_LOCAL, 'w', encoding='utf-8') as f:
            json.dump(list(banned_users), f, ensure_ascii=False, indent=2)
        with open(PENDING_FILE_LOCAL, 'w', encoding='utf-8') as f:
            json.dump({str(k): v for k, v in pending.items()}, f, ensure_ascii=False, indent=2)
    except Exception:
        logger.exception('Failed to persist state')

def persist_pending():
    persist_state()

def persist_bans():
    persist_state()

load_persisted_state()

# ----------------- Content -----------------
def load_content(path=CONTENT_FILE):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        logger.exception('Failed to load content file')
        return {}

CONTENT = load_content()
if not CONTENT:
    CONTENT = {'welcome_text': 'Hello!', 'main_menu': {}, 'tariffs_by_cat': {}, 'surveys': {}}

# ----------------- Keyboards / UI -----------------
def main_menu_kb():
    kb = types.ReplyKeyboardMarkup(row_width=2, resize_keyboard=True)
    menu = CONTENT.get('main_menu', {})
    order = ['1','2','3','4']
    labels = []
    for k in order:
        if k in menu and menu[k].get('label'):
            labels.append(menu[k]['label'])
    if len(labels) < 4:
        for v in menu.values():
            if v.get('label') not in labels:
                labels.append(v.get('label'))
            if len(labels) >= 4:
                break
    if len(labels) >= 2:
        kb.add(types.KeyboardButton(labels[0]), types.KeyboardButton(labels[1]))
    if len(labels) >= 4:
        kb.add(types.KeyboardButton(labels[2]), types.KeyboardButton(labels[3]))
    elif len(labels) == 3:
        kb.add(types.KeyboardButton(labels[2]))
    return kb

def build_subcategory_kb(catkey):
    cat = CONTENT.get('main_menu', {}).get(catkey, {})
    subs = cat.get('sub', [])
    ikb = types.InlineKeyboardMarkup()
    # special compact layouts
    if catkey == '1':
        first = subs[:10]
        for i in range(0, len(first), 2):
            if i+1 < len(first):
                ikb.row(types.InlineKeyboardButton(first[i]['label'], callback_data=f'sub|{catkey}|{i}'),
                        types.InlineKeyboardButton(first[i+1]['label'], callback_data=f'sub|{catkey}|{i+1}'))
            else:
                ikb.add(types.InlineKeyboardButton(first[i]['label'], callback_data=f'sub|{catkey}|{i}'))
        if len(subs) > 10:
            ikb.add(types.InlineKeyboardButton(subs[10]['label'], callback_data=f'sub|{catkey}|10'))
        ikb.add(types.InlineKeyboardButton('🔙 Back', callback_data='back_to_main'))
        return ikb
    if catkey == '2':
        first = subs[:6]
        for i in range(0, len(first), 2):
            if i+1 < len(first):
                ikb.row(types.InlineKeyboardButton(first[i]['label'], callback_data=f'sub|{catkey}|{i}'),
                        types.InlineKeyboardButton(first[i+1]['label'], callback_data=f'sub|{catkey}|{i+1}'))
            else:
                ikb.add(types.InlineKeyboardButton(first[i]['label'], callback_data=f'sub|{catkey}|{i}'))
        if len(subs) > 6:
            ikb.add(types.InlineKeyboardButton(subs[6]['label'], callback_data=f'sub|{catkey}|6'))
        ikb.add(types.InlineKeyboardButton('🔙 Back', callback_data='back_to_main'))
        return ikb
    # default layout
    for i in range(0, len(subs), 2):
        if i+1 < len(subs):
            ikb.row(types.InlineKeyboardButton(subs[i]['label'], callback_data=f'sub|{catkey}|{i}'),
                    types.InlineKeyboardButton(subs[i+1]['label'], callback_data=f'sub|{catkey}|{i+1}'))
        else:
            ikb.add(types.InlineKeyboardButton(subs[i]['label'], callback_data=f'sub|{catkey}|{i}'))
    ikb.add(types.InlineKeyboardButton('🔙 Back', callback_data='back_to_main'))
    return ikb

# ----------------- media sender helper -----------------
def try_send_media(chat_id, media, caption=None, reply_markup=None, disable_preview=False):
    \"\"\"Попытка отправить media разными методами.
    media может быть file_id или URL. Функция по очереди пробует:
      - send_video (если media выглядит как видео по расширению или если это file_id видео)
      - send_photo
      - send_document
      - send_animation
    Возвращает True при успешной отправке, иначе False.
    \"\"\"
    if not media:
        return False
    m = str(media).strip()
    is_url = m.startswith('http://') or m.startswith('https://')
    lower = m.lower()
    video_exts = ('.mp4', '.mov', '.webm', '.mkv', '.avi')
    animation_exts = ('.gif',)

    def _try(fn, *args, **kwargs):
        try:
            fn(*args, **kwargs)
            return True
        except Exception:
            logger.exception('Media send attempt failed for %s using %s', media, getattr(fn, '__name__', str(fn)))
            return False

    # If url ends with a video extension, try send_video first
    if is_url and any(lower.endswith(ext) for ext in video_exts):
        if _try(bot.send_video, chat_id, m, caption=caption or '', reply_markup=reply_markup):
            return True

    # If url looks like gif -> try animation
    if is_url and any(lower.endswith(ext) for ext in animation_exts):
        if _try(bot.send_animation, chat_id, m, caption=caption or '', reply_markup=reply_markup):
            return True

    # Try photo (covers photo file_id and image urls)
    if _try(bot.send_photo, chat_id, m, caption=caption or '', reply_markup=reply_markup):
        return True

    # Try video (works for video file_id or direct video urls)
    if _try(bot.send_video, chat_id, m, caption=caption or '', reply_markup=reply_markup):
        return True

    # Try document (generic fallback)
    if _try(bot.send_document, chat_id, m, caption=caption or '', reply_markup=reply_markup):
        return True

    # Try animation as last fallback
    if _try(bot.send_animation, chat_id, m, caption=caption or '', reply_markup=reply_markup):
        return True

    logger.warning('All attempts to send media failed for %s to chat %s', media, chat_id)
    return False

def admin_buttons_for(user_id: int) -> types.InlineKeyboardMarkup:
    ikb = types.InlineKeyboardMarkup()
    ikb.add(types.InlineKeyboardButton('Reply', callback_data=f'admin_reply|{user_id}'))
    ikb.add(types.InlineKeyboardButton('Ban', callback_data=f'admin_ban|{user_id}'))
    ikb.add(types.InlineKeyboardButton('Unban', callback_data=f'admin_unban|{user_id}'))
    ikb.add(types.InlineKeyboardButton('Profile', callback_data=f'admin_profile|{user_id}'))
    return ikb

# ----------------- Utilities -----------------
def is_banned(user_id: int) -> bool:
    return int(user_id) in banned_users

def ban_user(user_id: int):
    banned_users.add(int(user_id))
    persist_bans()

def unban_user(user_id: int):
    banned_users.discard(int(user_id))
    persist_bans()

def is_subscribed(user_id):
    target = None
    if CHANNEL_ID:
        target = CHANNEL_ID
    elif CHANNEL_USERNAME:
        target = CHANNEL_USERNAME
    else:
        logger.warning('No channel configured for subscription check')
        return False
    try:
        res = bot.get_chat_member(target, user_id)
        return getattr(res, 'status', None) in ('member', 'administrator', 'creator', 'restricted')
    except Exception:
        logger.exception('get_chat_member failed')
        return False

# ----------------- Handlers -----------------
@bot.message_handler(commands=['start'])
def start_handler(message):
    uid = message.from_user.id
    allowed, reason = check_rate(uid)
    if not allowed:
        bot.send_message(uid, reason)
        return
    if is_banned(uid):
        bot.send_message(uid, '🚫 You have been blocked and cannot use the bot')
        return
    if not is_subscribed(uid):
        kb = types.InlineKeyboardMarkup()
        if CHANNEL_USERNAME:
            kb.add(types.InlineKeyboardButton('🏃‍➡️ Go to the channel', url=f'https://t.me/{CHANNEL_USERNAME.lstrip("@")}'))
        kb.add(types.InlineKeyboardButton('✅ Check subscription', callback_data='check_sub'))
        bot.send_message(uid, CONTENT.get('welcome_text', 'Hello!'), reply_markup=kb)
        return
    bot.send_message(uid, f"Hello, {message.from_user.first_name}! 👇 Use the menu below to continue:", reply_markup=main_menu_kb())

@bot.message_handler(commands=['myid'])
def cmd_myid(m):
    bot.send_message(m.chat.id, f'Your chat id: {m.chat.id}')

@bot.message_handler(commands=['reply'])
def cmd_reply(m):
    if m.chat.id not in ADMIN_IDS:
        bot.send_message(m.chat.id, '🚫 No access')
        return
    parts = (m.text or '').split()
    if len(parts) < 2:
        bot.send_message(m.chat.id, 'Using: /reply <user_id>')
        return
    try:
        uid = int(parts[1])
        pending[int(m.chat.id)] = uid
        persist_pending()
        bot.send_message(m.chat.id, f'⏳ Ожидаю ответ — следующее ваше сообщение будет отправлено пользователю {uid}')
    except Exception:
        bot.send_message(m.chat.id, 'Incorrect user_id')

@bot.message_handler(commands=['admin'])
def cmd_admin(m):
    if m.chat.id not in ADMIN_IDS:
        bot.send_message(m.chat.id, '🚫 No access')
        return
    if not recent_messages:
        bot.send_message(m.chat.id, '📭 No recent messages')
        return
    bot.send_message(m.chat.id, f'Last {min(len(recent_messages),10)} messages:')
    for rec in list(recent_messages)[-10:]:
        preview = rec.get('preview')
        uid = rec.get('user_id')
        try:
            bot.send_message(m.chat.id, f'From {rec.get("username") or "no_name"} (id: {uid})\n{preview}', reply_markup=admin_buttons_for(uid))
        except Exception:
            logger.exception('Failed to send admin preview')

@bot.callback_query_handler(func=lambda c: c.data == 'check_sub')
def cb_check(c):
    uid = c.from_user.id
    allowed, reason = check_rate(uid)
    if not allowed:
        bot.answer_callback_query(c.id, reason)
        return
    if is_banned(uid):
        bot.answer_callback_query(c.id, '🚫 You are blocked')
        return
    if is_subscribed(uid):
        bot.answer_callback_query(c.id, '✅ Access granted')
        bot.send_message(uid, '👇 Use the menu below to continue:', reply_markup=main_menu_kb())
    else:
        bot.answer_callback_query(c.id, '❌ You are still not subscribed')

@bot.message_handler(func=lambda m: m.text in [v.get('label') for v in CONTENT.get('main_menu', {}).values()])
def cat_handler(m):
    uid = m.from_user.id
    allowed, reason = check_rate(uid)
    if not allowed:
        bot.send_message(uid, reason)
        return
    if is_banned(uid):
        bot.send_message(uid, '🚫 You are blocked and cannot use the bot')
        return
    key = None
    for k, v in CONTENT.get('main_menu', {}).items():
        if m.text == v.get('label'):
            key = k
            cat = v
            break
    if not key:
        return
    ikb = build_subcategory_kb(key)
    caption = (cat.get('text','') or '').strip()
    if caption:
        caption = f"{caption}\n\n🔻 Choose a tariff 🔻"
    else:
        caption = '🔻 Choose a tariff 🔻'
    media = cat.get('media')
    if media:
        sent = try_send_media(uid, media, caption=caption, reply_markup=ikb)
        if not sent:
            bot.send_message(uid, caption, reply_markup=ikb)
    else:
        bot.send_message(uid, caption, reply_markup=ikb)

# Admin reply handler (pending)
@bot.message_handler(func=lambda m: m.chat.id in ADMIN_IDS and int(m.chat.id) in pending, content_types=['text','photo','video','document'])
def admin_send_reply(m):
    admin_id = m.chat.id
    target = pending.get(int(admin_id))
    if not target:
        bot.send_message(admin_id, 'Нет активного пользователя для ответа. Нажмите Reply рядом с сообщением пользователя или используйте /reply <user_id>.')
        return
    try:
        if m.content_type == 'text':
            bot.send_message(target, m.text)
        elif m.content_type == 'photo':
            file_id = m.photo[-1].file_id
            bot.send_photo(target, file_id, caption=(m.caption or ''))
        elif m.content_type == 'video':
            bot.send_video(target, m.video.file_id, caption=(m.caption or ''))
        elif m.content_type == 'document':
            bot.send_document(target, m.document.file_id, caption=(m.caption or ''))
        bot.send_message(admin_id, f'Ответ отправлен пользователю {target}.')
    except Exception:
        logger.exception('Failed to send reply to user')
        bot.send_message(admin_id, 'Ошибка при отправке ответа.')
    pending.pop(int(admin_id), None)
    persist_pending()

@bot.message_handler(func=lambda m: user_state.get(m.from_user.id) and user_state.get(m.from_user.id).get('survey'), content_types=['text'])
def survey_catcher(m):
    uid = m.from_user.id
    allowed, reason = check_rate(uid)
    if not allowed:
        try:
            bot.send_message(uid, reason)
        except Exception:
            pass
        return
    st = user_state.get(uid)
    if st and st.get('survey'):
        survey = st['survey']
        qidx = survey.get('qindex', 0)
        answers = survey.get('answers', [])
        answers.append(m.text)
        survey['answers'] = answers
        survey['qindex'] = qidx + 1
        catkey = survey.get('catkey')
        questions = CONTENT.get('surveys', {}).get(catkey, [])
        if survey['qindex'] < len(questions):
            bot.send_message(uid, questions[survey['qindex']])
        else:
            bot.send_message(uid, '✅ Thank you! Your answers have been accepted. We will contact you soon.')
            summary = 'Survey results:\n'
            for i, a in enumerate(survey['answers'], start=1):
                summary += f"{i}. {a}\n"
            for aid in ADMIN_IDS:
                try:
                    bot.send_message(aid, f"Опрос от пользователя {uid}:\n{summary}", reply_markup=admin_buttons_for(uid))
                except Exception:
                    logger.exception('Failed to send survey results to admin %s', aid)
            user_state.pop(uid, None)
    return

@bot.message_handler(func=lambda m: True, content_types=['text','photo','video','document','animation','voice','video_note'])
def forward_to_admins(m):
    uid = m.from_user.id
    # preserve early returns / guards
    if uid in ADMIN_IDS:
        return
    text = m.text or ''
    labels = [v.get('label') for v in CONTENT.get('main_menu', {}).values()]
    if text and text in labels:
        return
    if is_banned(uid):
        try:
            bot.send_message(uid, '🚫 You have been blocked and cannot use the bot')
        except Exception:
            pass
        return
    allowed, reason = check_rate(uid)
    if not allowed:
        try:
            bot.send_message(uid, reason)
        except Exception:
            pass
        return

    uname = m.from_user.username or m.from_user.first_name or ''
    caption = f"Сообщение от @{uname} (id: {uid}):\n"
    try:
        # TEXT
        if m.content_type == 'text':
            body = m.text
            preview = body[:300]
            for aid in ADMIN_IDS:
                try:
                    bot.send_message(aid, caption + '\n' + body, reply_markup=admin_buttons_for(uid))
                except Exception:
                    logger.exception('Failed to forward text to admin %s', aid)
            rec = {'user_id': uid, 'username': uname, 'type': 'text', 'preview': preview, 'ts': int(time.time())}
            # confirmation to user
            try:
                bot.send_message(uid, 'Сообщение получено. Спасибо.')
            except Exception:
                logger.exception('Failed to confirm text to user %s', uid)
            logger.info('Forwarded text from %s id=%s', uname, uid)

        # PHOTO
        elif m.content_type == 'photo':
            file_id = m.photo[-1].file_id
            preview = (m.caption or '')[:300]
            # immediate feedback to user with file_id (requested behavior)
            try:
                bot.send_message(uid, f'Фото получено. file_id: {file_id}')
            except Exception:
                logger.exception('Failed to send immediate photo file_id to user %s', uid)
            # forward to admins
            for aid in ADMIN_IDS:
                try:
                    bot.send_photo(aid, file_id, caption=caption + (m.caption or ''), reply_markup=admin_buttons_for(uid))
                except Exception:
                    logger.exception('Failed to forward photo to admin %s', aid)
            rec = {'user_id': uid, 'username': uname, 'type': 'photo', 'preview': preview, 'file_id': file_id, 'ts': int(time.time())}
            logger.info('Received photo from user=%s id=%s file_id=%s', uname, uid, file_id)

        # VIDEO
        elif m.content_type == 'video':
            file_id = m.video.file_id
            preview = (m.caption or '')[:300]
            try:
                bot.send_message(uid, f'Видео получено. file_id: {file_id}')
            except Exception:
                logger.exception('Failed to send immediate video file_id to user %s', uid)
            for aid in ADMIN_IDS:
                try:
                    bot.send_video(aid, file_id, caption=caption + (m.caption or ''), reply_markup=admin_buttons_for(uid))
                except Exception:
                    logger.exception('Failed to forward video to admin %s', aid)
            rec = {'user_id': uid, 'username': uname, 'type': 'video', 'preview': preview, 'file_id': file_id, 'ts': int(time.time())}
            logger.info('Received video from user=%s id=%s file_id=%s', uname, uid, file_id)

        # DOCUMENT
        elif m.content_type == 'document':
            file_id = m.document.file_id
            preview = (m.caption or '')[:300]
            try:
                bot.send_message(uid, f'Файл получен. file_id: {file_id}')
            except Exception:
                logger.exception('Failed to send immediate document file_id to user %s', uid)
            for aid in ADMIN_IDS:
                try:
                    bot.send_document(aid, file_id, caption=caption + (m.caption or ''), reply_markup=admin_buttons_for(uid))
                except Exception:
                    logger.exception('Failed to forward document to admin %s', aid)
            rec = {'user_id': uid, 'username': uname, 'type': 'document', 'preview': preview, 'file_id': file_id, 'ts': int(time.time())}
            logger.info('Received document from user=%s id=%s file_id=%s', uname, uid, file_id)

        # ANIMATION (GIF)
        elif m.content_type == 'animation':
            file_id = m.animation.file_id
            try:
                bot.send_message(uid, f'Анимация получена. file_id: {file_id}')
            except Exception:
                logger.exception('Failed to send immediate animation file_id to user %s', uid)
            for aid in ADMIN_IDS:
                try:
                    bot.send_animation(aid, file_id, caption=caption + (m.caption or ''), reply_markup=admin_buttons_for(uid))
                except Exception:
                    logger.exception('Failed to forward animation to admin %s', aid)
            rec = {'user_id': uid, 'username': uname, 'type': 'animation', 'file_id': file_id, 'ts': int(time.time())}
            logger.info('Received animation from user=%s id=%s file_id=%s', uname, uid, file_id)

        # VOICE
        elif m.content_type == 'voice':
            file_id = m.voice.file_id
            try:
                bot.send_message(uid, f'Голосовое сообщение получено. file_id: {file_id}')
            except Exception:
                logger.exception('Failed to send immediate voice file_id to user %s', uid)
            for aid in ADMIN_IDS:
                try:
                    bot.send_voice(aid, file_id, caption=caption + (m.caption or ''), reply_markup=admin_buttons_for(uid))
                except Exception:
                    logger.exception('Failed to forward voice to admin %s', aid)
            rec = {'user_id': uid, 'username': uname, 'type': 'voice', 'file_id': file_id, 'ts': int(time.time())}
            logger.info('Received voice from user=%s id=%s file_id=%s', uname, uid, file_id)

        # VIDEO NOTE
        elif m.content_type == 'video_note':
            file_id = m.video_note.file_id
            try:
                bot.send_message(uid, f'Видео-заметка получена. file_id: {file_id}')
            except Exception:
                logger.exception('Failed to send immediate video_note file_id to user %s', uid)
            for aid in ADMIN_IDS:
                try:
                    bot.send_video_note(aid, file_id, reply_markup=admin_buttons_for(uid))
                except Exception:
                    logger.exception('Failed to forward video_note to admin %s', aid)
            rec = {'user_id': uid, 'username': uname, 'type': 'video_note', 'file_id': file_id, 'ts': int(time.time())}
            logger.info('Received video_note from user=%s id=%s file_id=%s', uname, uid, file_id)

        else:
            rec = {'user_id': uid, 'username': uname, 'type': 'other', 'preview': '', 'ts': int(time.time())}

        # store recent
        recent_messages.append(rec)
        if len(recent_messages) > RECENT_MAX:
            recent_messages.pop(0)
    except Exception:
        logger.exception('forward_to_admins failed')
    return

@bot.callback_query_handler(func=lambda c: True)
def all_cb(c):
    data = c.data
    user_id = c.from_user.id
    # admin actions
    if data.startswith('admin_reply|'):
        if user_id not in ADMIN_IDS:
            bot.answer_callback_query(c.id, '🚫 No access')
            return
        target = int(data.split('|',1)[1])
        pending[int(user_id)] = int(target)
        persist_pending()
        bot.answer_callback_query(c.id, '⏳ Ожидаю ответ — следующее ваше сообщение будет отправлено пользователю')
        return
    if data.startswith('admin_ban|'):
        if user_id not in ADMIN_IDS:
            bot.answer_callback_query(c.id, '🚫 No access')
            return
        target = int(data.split('|',1)[1])
        ban_user(target)
        bot.answer_callback_query(c.id, f'Пользователь {target} заблокирован')
        try:
            bot.send_message(target, '🚫 You have been blocked and cannot use the bot')
        except Exception:
            pass
        return
    if data.startswith('admin_unban|'):
        if user_id not in ADMIN_IDS:
            bot.answer_callback_query(c.id, '🚫 No access')
            return
        target = int(data.split('|',1)[1])
        unban_user(target)
        bot.answer_callback_query(c.id, f'Пользователь {target} разаблокирован')
        return
    if data.startswith('admin_profile|'):
        if user_id not in ADMIN_IDS:
            bot.answer_callback_query(c.id, '🚫 No access')
            return
        target = int(data.split('|',1)[1])
        try:
            info = bot.get_chat(target)
            text = f"Profile: id={info.id}, type={info.type}, title={getattr(info, 'title', '')}, username={getattr(info, 'username', '')}"
        except Exception:
            text = f"Не удалось получить информацию о пользователе {target}"
        bot.answer_callback_query(c.id)
        bot.send_message(user_id, text)
        return

    # navigation
    if data == 'back_to_main':
        bot.answer_callback_query(c.id)
        bot.send_message(user_id, 'Main Menu:', reply_markup=main_menu_kb())
        return
    if data.startswith('back_to_sub|'):
        parts = data.split('|')
        if len(parts) >= 2:
            catkey = parts[1]
            bot.answer_callback_query(c.id)
            cat = CONTENT.get('main_menu', {}).get(catkey, {})
            ikb = build_subcategory_kb(catkey)
            caption = (cat.get('text', '') or '').strip()
            if caption:
                caption = f"{caption}\n\n🔻 Choose a tariff 🔻"
            else:
                caption = '🔻 Choose a tariff 🔻'
            media = cat.get('media')
            if media:
                try:
                    bot.send_photo(user_id, media, caption=caption, reply_markup=ikb)
                except Exception:
                    bot.send_message(user_id, caption, reply_markup=ikb)
            else:
                bot.send_message(user_id, caption, reply_markup=ikb)
        return

    if data.startswith('sub|'):
        try:
            _, catkey, idx = data.split('|')
            idx = int(idx)
        except Exception:
            bot.answer_callback_query(c.id, '❌ Invalid command')
            return
        cat = CONTENT.get('main_menu', {}).get(catkey, {})
        subs = cat.get('sub', [])
        if idx < 0 or idx >= len(subs):
            bot.answer_callback_query(c.id, '❌ Invalid tariff')
            return
        sub = subs[idx]
        sublabel = (sub.get('label', '') or '').strip().lower()
        # --- special flows for categories 3 and 4 ---
        # Category 4: explicit "leave message" action OR legacy label starting with 'оставить'
        is_leave = (sub.get('action') == 'leave_message') or sublabel.startswith('оставить')
        if catkey == '4' and is_leave:
            # Show a simple static message; do NOT show tariffs or "🔙 Back" menu here.
            bot.answer_callback_query(c.id)
            bot.send_message(user_id, '❌ You must purchase something before leaving a review')
            return

        # Category 3: survey flow.
        # Support two ways to mark subcategory as survey:
        #  - preferred: add "action": "survey" in content.json for the subcategory
        #  - legacy: keep sub label starting with "admin" (case-insensitive)
        is_survey = (sub.get('action') == 'survey') or sublabel.startswith('admin')
        if catkey == '3' and is_survey:
            bot.answer_callback_query(c.id)
            ikb = types.InlineKeyboardMarkup()
            ikb.add(types.InlineKeyboardButton('📝Apply', callback_data=f'start_survey|{catkey}|{idx}'))
            ikb.add(types.InlineKeyboardButton('🔙 Back', callback_data=f'back_to_sub|{catkey}'))
            caption = (sub.get('text', '') or '').strip()
            if caption:
                caption = f"{caption}\n\nClick 📝Apply to start the survey"
            else:
                caption = 'Click 📝Apply to start the survey'
            media = sub.get('media')
            if media:
                sent = try_send_media(user_id, media, caption=caption, reply_markup=ikb)
                if not sent:
                    bot.send_message(user_id, caption, reply_markup=ikb)
            else:
                bot.send_message(user_id, caption, reply_markup=ikb)
            return

        # ---- tariffs selection: support per-subcategory tariffs ----
        t_by_cat = CONTENT.get('tariffs_by_cat', {})
        tariffs = []
        cat_tariffs = t_by_cat.get(catkey)
        if isinstance(cat_tariffs, dict):
            tariffs = cat_tariffs.get(str(idx), [])
        elif isinstance(cat_tariffs, list):
            tariffs = cat_tariffs
        else:
            tariffs = t_by_cat.get('default', [])

        ikb = types.InlineKeyboardMarkup()
        for t in tariffs:
            safe_t = str(t).replace('|', '/')
            ikb.add(types.InlineKeyboardButton(str(t), callback_data=f'tariff|{catkey}|{idx}|{safe_t}'))
        ikb.add(types.InlineKeyboardButton('🔙 Back', callback_data=f'back_to_sub|{catkey}'))

        caption = (sub.get('text', '') or '').strip()
        if caption:
            caption = f"{caption}\n\n🔻 Choose a tariff 🔻"
        else:
            caption = '🔻 Choose a tariff 🔻'
        media = sub.get('media')
        if media:
            sent = try_send_media(user_id, media, caption=caption, reply_markup=ikb)
            if not sent:
                bot.send_message(user_id, caption, reply_markup=ikb)
        else:
            bot.send_message(user_id, caption, reply_markup=ikb)
        bot.answer_callback_query(c.id)
        return

    if data.startswith('start_survey|'):
        parts = data.split('|', 2)
        if len(parts) < 3:
            bot.answer_callback_query(c.id, '⚠️Survey launch error')
            return
        _, catkey, subidx = parts
        bot.answer_callback_query(c.id)
        questions = CONTENT.get('surveys', {}).get(catkey, [])
        if not questions:
            bot.send_message(user_id, '⚙️The survey is unavailable')
            return
        user_state[user_id] = {'survey': {'catkey': catkey, 'qindex': 0, 'answers': []}}
        bot.send_message(user_id, questions[0])
        return

    if data.startswith('tariff|'):
        parts = data.split('|', 3)
        if len(parts) < 4:
            bot.answer_callback_query(c.id, '⚠️Data error')
            return
        _, catkey, subidx, tariff_name = parts
        bot.answer_callback_query(c.id)
        pay_kb = types.InlineKeyboardMarkup()
        pay_kb.add(types.InlineKeyboardButton('💸Payment', callback_data=f'pay|{catkey}|{subidx}|{tariff_name}'))
        pay_kb.add(types.InlineKeyboardButton('🔙 Back', callback_data=f'back_to_sub|{catkey}'))
        sub_media = None
        try:
            si = int(subidx)
            subs = CONTENT.get('main_menu', {}).get(catkey, {}).get('sub', [])
            if 0 <= si < len(subs):
                sub_media = subs[si].get('media')
        except Exception:
            sub_media = None
        caption = f"You have chosen: {tariff_name}.\n\n💸 To pay, click the button below"
        if sub_media:
            sent = try_send_media(user_id, sub_media, caption=caption, reply_markup=pay_kb)
            if not sent:
                bot.send_message(user_id, caption, reply_markup=pay_kb)
        else:
            bot.send_message(user_id, caption, reply_markup=pay_kb)
        return

        if data.startswith('pay|'):
            bot.answer_callback_query(c.id)
            # Получаем текст инструкции из content.json (если есть)
            pay_text = CONTENT.get('pay_text', '') or ''
            # Ссылка на инструкцию: переменная окружения имеет приоритет, затем content.json
            pay_link = (TELEGRAPH_PAY_LINK and TELEGRAPH_PAY_LINK.strip()) or (CONTENT.get('pay_link') or '').strip()
            # Собираем сообщение: текст (если есть) + ссылка (если есть)
            parts = []
            if pay_text:
                parts.append(pay_text.strip())
            if pay_link:
                # если ссылка не входит в pay_text, добавим её отдельной строкой
                parts.append(pay_link)
            message = '\n\n'.join(parts) if parts else '💸Payment'
            # Если есть ссылка — добавим кнопку с переходом
            if pay_link:
                ikb = types.InlineKeyboardMarkup()
                ikb.add(types.InlineKeyboardButton('💸Payment', url=pay_link))
                bot.send_message(user_id, message, reply_markup=ikb, disable_web_page_preview=False)
            else:
                bot.send_message(user_id, message)
            return

    bot.answer_callback_query(c.id, '⚠️Unknown command')

# Admin commands for ban/unban/list
@bot.message_handler(func=lambda m: m.chat.id in ADMIN_IDS and m.text and m.text.startswith('/ban'))
def cmd_ban(m):
    parts = m.text.split()
    if len(parts) < 2:
        bot.send_message(m.chat.id, 'Использование: /ban <user_id>')
        return
    try:
        uid = int(parts[1])
        ban_user(uid)
        bot.send_message(m.chat.id, f'Пользователь {uid} заблокирован')
    except Exception:
        bot.send_message(m.chat.id, 'Ошибка: неверный user_id')

@bot.message_handler(func=lambda m: m.chat.id in ADMIN_IDS and m.text and m.text.startswith('/unban'))
def cmd_unban(m):
    parts = m.text.split()
    if len(parts) < 2:
        bot.send_message(m.chat.id, 'Использование: /unban <user_id>')
        return
    try:
        uid = int(parts[1])
        unban_user(uid)
        bot.send_message(m.chat.id, f'Пользователь {uid} разблокирован')
    except Exception:
        bot.send_message(m.chat.id, 'Ошибка: неверный user_id')

@bot.message_handler(func=lambda m: m.chat.id in ADMIN_IDS and m.text and m.text.startswith('/listbans'))
def cmd_listbans(m):
    if not banned_users:
        bot.send_message(m.chat.id, 'Список заблокированных пуст')
        return
    bot.send_message(m.chat.id, 'Заблокированные пользователи:\n' + '\n'.join(str(x) for x in banned_users))

# Flask endpoints
@app.route('/', methods=['GET'])
def index():
    return 'OK', 200

@app.route(WEBHOOK_PATH, methods=['POST'])
def webhook():
    # Fast-path: читаем raw JSON, пытаемся сразу достать file_id и отправить его пользователю
    if request.headers.get('content-type') == 'application/json':
        raw = request.get_data().decode('utf-8')
        try:
            data = json.loads(raw)
        except Exception:
            data = None

        # Попытка быстро извлечь file_id из Update и тут же отправить пользователю
        try:
            if isinstance(data, dict):
                msg = data.get('message') or data.get('edited_message') or data.get('channel_post')
                if msg and isinstance(msg, dict):
                    chat = msg.get('chat') or {}
                    chat_id = chat.get('id') or (msg.get('from') or {}).get('id')
                    file_id = None
                    # Проверяем распространенные поля
                    if 'photo' in msg and isinstance(msg['photo'], list) and msg['photo']:
                        file_id = msg['photo'][-1].get('file_id')
                    elif 'video' in msg and isinstance(msg['video'], dict):
                        file_id = msg['video'].get('file_id')
                    elif 'document' in msg and isinstance(msg['document'], dict):
                        file_id = msg['document'].get('file_id')
                    elif 'animation' in msg and isinstance(msg['animation'], dict):
                        file_id = msg['animation'].get('file_id')
                    elif 'voice' in msg and isinstance(msg['voice'], dict):
                        file_id = msg['voice'].get('file_id')
                    elif 'video_note' in msg and isinstance(msg['video_note'], dict):
                        file_id = msg['video_note'].get('file_id')

                    if file_id and chat_id:
                        # отправляем пользователю сразу — гарантированно увидите file_id в чате
                        try:
                            bot.send_message(chat_id, f'file_id: {file_id}')
                        except Exception:
                            logger.exception('Failed to send immediate file_id to user %s', chat_id)
                        # логируем на сервере
                        logger.info('Webhook raw: received file_id=%s from chat=%s', file_id, chat_id)
        except Exception:
            logger.exception('Failed to parse raw webhook update for quick file_id')

        # Пропускаем Update в telebot, чтобы существующие хендлеры работали как раньше
        try:
            upd = telebot.types.Update.de_json(raw)
            bot.process_new_updates([upd])
            return '', 200
        except Exception:
            logger.exception('Failed to process update via telebot')
            return '', 200
    abort(403)

def setup_webhook():
    if WEBHOOK_URL:
        try:
            bot.remove_webhook()
        except Exception:
            pass
        try:
            bot.set_webhook(url=WEBHOOK_URL)
            logger.info('Webhook set to %s', WEBHOOK_URL)
        except Exception:
            logger.exception('Failed to set webhook')

setup_webhook()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)))