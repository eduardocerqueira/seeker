#date: 2026-02-25T17:46:31Z
#url: https://api.github.com/gists/9f669d295e98d474815cceb60684df99
#owner: https://api.github.com/users/KozlovAleksei

import os
import logging
from logging.handlers import RotatingFileHandler
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Относительные пути
LAST_DIR = os.path.join(os.path.dirname(__file__), 'last')
LOG_FILE = os.path.join(os.path.dirname(__file__), 'txt2pdf.log')
LOG_MAX_SIZE = 5 * 1024 * 1024  # 5 MB
LOG_BACKUP_COUNT = 3  # Количество резервных копий логов

def setup_logging():
    """Настраивает логирование с ротацией логов."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Обработчик для записи в файл с ротацией
    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=LOG_MAX_SIZE,
        backupCount=LOG_BACKUP_COUNT,
        encoding='utf-8'
    )
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    # Обработчик для вывода в консоль
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    # Добавление обработчиков
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

def log_error(message):
    """Записывает сообщение об ошибке в лог."""
    logging.error(message)

def log_info(message):
    """Записывает информационное сообщение в лог."""
    logging.info(message)

def convert_txt_to_pdf(txt_path, pdf_path):
    """Конвертирует текстовый файл в PDF."""
    try:
        c = canvas.Canvas(pdf_path, pagesize=letter)
        with open(txt_path, 'r', encoding='utf-8') as txt_file:
            text = txt_file.read()

        # Разбиваем текст на строки и добавляем на страницу
        text_lines = text.split('\n')
        y_position = 750  # Начальная позиция по вертикали
        for line in text_lines:
            c.drawString(50, y_position, line)
            y_position -= 15  # Отступ между строками
            if y_position < 50:  # Если достигли низа страницы
                c.showPage()  # Создаём новую страницу
                y_position = 750  # Сбрасываем позицию

        c.save()
        log_info(f"Файл {txt_path} успешно конвертирован в {pdf_path}")
        return True

    except Exception as e:
        log_error(f"Ошибка при конвертации {txt_path}: {e}")
        return False

def process_txts_in_last():
    """Обрабатывает все текстовые файлы с префиксом 2_ в папке last."""
    if not os.path.exists(LAST_DIR):
        log_error(f"Папка {LAST_DIR} не найдена.")
        return

    txt_files = [f for f in os.listdir(LAST_DIR) if f.startswith('2_') and f.lower().endswith('.txt')]

    if not txt_files:
        log_info("Текстовые файлы с префиксом 2_ в папке last не найдены.")
        return

    for txt_file in txt_files:
        txt_path = os.path.join(LAST_DIR, txt_file)

        # Формируем имя для PDF-файла с префиксом 3_
        pdf_filename = f"3_{txt_file.replace('2_', '').replace('.txt', '.pdf')}"
        pdf_path = os.path.join(LAST_DIR, pdf_filename)

        convert_txt_to_pdf(txt_path, pdf_path)

def main():
    """Основная функция: настройка логирования и обработка текстовых файлов."""
    setup_logging()
    log_info("Запуск txt2pdf.exe")
    process_txts_in_last()

if __name__ == "__main__":
    main()
