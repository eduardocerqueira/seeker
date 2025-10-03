#date: 2025-10-03T16:59:15Z
#url: https://api.github.com/gists/461779c26081765be63d11efe553ceb0
#owner: https://api.github.com/users/mgaitan

# /// script
# dependencies = [
#     "python-telegram-bot>=20.0",
# ]
# ///

import os
import mimetypes
from pathlib import Path
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters

TMP_DIR = Path("/tmp")
SCRIPT_DIR= Path(__file__).resolve().parent
AUDIO_SAMPLE = SCRIPT_DIR / "audio.mp3"


def _pick_extension(file_name: str | None, mime_type: str | None) -> str:
    """Return a safe extension for the incoming audio clip."""
    if file_name:
        suffix = Path(file_name).suffix
        if suffix:
            return suffix
    if mime_type:
        guessed = mimetypes.guess_extension(mime_type)
        if guessed:
            return guessed
    return ".bin"


async def handle_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Reply to /start with a short greeting."""
    message = update.effective_message
    if message:
        await message.reply_text("hola, soy tu bot")


async def handle_audio_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send the bundled sample when the user issues /audio."""
    message = update.effective_message
    # Read the local sample and send it back as an audio attachment.
    with AUDIO_SAMPLE.open("rb") as audio_stream:
        await message.reply_audio(audio=audio_stream)


async def handle_incoming_audio(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Persist voice notes or audio files to /tmp and confirm receipt."""
    message = update.effective_message
    if not message:
        return

    attachment = message.voice or message.audio
    if not attachment:
        return

    extension = _pick_extension(
        getattr(attachment, "file_name", None),
        getattr(attachment, "mime_type", None),
    )
    file_name = (
        f"chat{message.chat_id}_msg{message.message_id}_"
        f"{attachment.file_unique_id}{extension}"
    )
    destination = TMP_DIR / file_name

    TMP_DIR.mkdir(parents=True, exist_ok=True)
    telegram_file = await attachment.get_file()
    await telegram_file.download_to_drive(custom_path=destination)

    await message.reply_text("recibi tu mensaje y lo guardé correctamente")


def main() -> None:
    token = "**********"

    app = "**********"
    app.add_handler(CommandHandler("start", handle_start))
    app.add_handler(CommandHandler("audio", handle_audio_command))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_incoming_audio))

    app.run_polling()


if __name__ == "__main__":
    main()
olling()


if __name__ == "__main__":
    main()
