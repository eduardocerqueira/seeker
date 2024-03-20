#date: 2024-03-20T17:04:57Z
#url: https://api.github.com/gists/3a6d4348be3d45a4f605b64a91418b62
#owner: https://api.github.com/users/anon25519

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QDesktopWidget, QMainWindow, QFileDialog, QVBoxLayout, QFormLayout, QWidget, QLabel, QLineEdit, QTextEdit, QPushButton, QProgressBar, QSizePolicy, QErrorMessage
import os
import sys
import mmap
import hashlib
import shutil

# Преобразование DOS-путей в UNC-пути для обхода ограничения в 260 символов на путь к файлу
# https://stackoverflow.com/questions/36219317/pathname-too-long-to-open
def uncPath(dosPath):
    if dosPath.startswith('\\\\?\\'):
        return dosPath
    return '\\\\?\\' + os.path.abspath(dosPath)

def unc_isfile(path, *args, **kwargs):
    return os.path.isfile(uncPath(path), *args, **kwargs)

def unc_getsize(path, *args, **kwargs):
    return os.path.getsize(uncPath(path), *args, **kwargs)

def unc_copyfile(src, dst, *args, **kwargs):
    return shutil.copyfile(uncPath(src), uncPath(dst), *args, **kwargs)

_open = open
def open(path, *args, **kwargs):
    return _open(uncPath(path), *args, **kwargs)

TARGET_EXTENSIONS = ('.jpg', '.jpeg', '.jpe', '.png', '.gif', '.webp', '.webm', '.mp4')

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Поиск файлов по хэшу")
        self.resize(800, 600)
        rect = self.frameGeometry()
        rect.moveCenter(QDesktopWidget().availableGeometry().center())
        self.move(rect.topLeft())

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        self.status_label = QLabel()
        self.status_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.layout.addWidget(self.status_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.progress_bar)
        
        self.log_text_edit = QTextEdit()
        self.log_text_edit.setReadOnly(True)
        self.layout.addWidget(self.log_text_edit)
        
        self.pause_button = QPushButton("Пауза")
        self.pause_button.clicked.connect(self.pause_search)
        self.layout.addWidget(self.pause_button)
        self.pause_button.hide()
        self.is_paused = False

    def pause_search(self):
        if self.is_paused:
            self.pause_button.setText("Пауза")
            self.is_paused = False
        else:
            self.pause_button.setText("Возобновить")
            self.is_paused = True

    def search_files(self, dir_to_scan, dir_for_results):
        self.status_label.setText("Загрузка списка искомых файлов...")
        hashes_file = os.path.join(sys.path[0], "hashes.bin")
        self.progress_bar.setMaximum(unc_getsize(hashes_file))
        chunk_size = 1024 * 1024
        target_hashes = set()
        with open(hashes_file, 'rb') as f:
            mm = mmap.mmap(f.fileno(), 0, access = mmap.ACCESS_READ)
            while True:
                buffer = mm.read(chunk_size)
                if not buffer:
                    break
                target_hashes.update([buffer[i:i+16] for i in range(0, len(buffer), 16)])
                position = mm.tell()
                if position % chunk_size == 0:
                    self.progress_bar.setValue(position)
                    QApplication.processEvents()

        self.pause_button.show()
        self.status_label.setText("Поиск локальных файлов...")
        QApplication.processEvents()
        all_files = []
        for root, dirs, files in os.walk(dir_to_scan):
            for file in files:
                if file.lower().endswith(TARGET_EXTENSIONS):
                    all_files.append(os.path.join(root, file))
                if self.is_paused:
                    while self.is_paused:
                        QApplication.processEvents()
        
        self.progress_bar.setMaximum(len(all_files))
        found_files = 0
        saved_files = 0
        for index, filename in enumerate(all_files):
            if unc_getsize(filename) == 0:
                continue

            self.status_label.setText("Проверяется файл: " + filename)
            if self.is_paused:
                while self.is_paused:
                    QApplication.processEvents()
            else:
                QApplication.processEvents()    
            
            h = hashlib.md5()
            with open(filename, 'rb') as f:
                mm = mmap.mmap(f.fileno(), 0, access = mmap.ACCESS_READ)
                while True:
                    buffer = mm.read(chunk_size)
                    if not buffer:
                        break
                    h.update(buffer)
            if h.digest() in target_hashes:
                found_files += 1
                self.log_text_edit.append(h.hexdigest().upper() + " = " + filename)
                if dir_for_results != '.':
                    file_name, file_ext = os.path.splitext(filename)
                    dest_filename = os.path.join(dir_for_results, h.hexdigest()) + file_ext
                    if not unc_isfile(dest_filename):
                        self.log_text_edit.append(f"копирование \"{filename}\" в \"{dest_filename}\"")
                        try:
                            unc_copyfile(filename, dest_filename)
                        except shutil.Error as err:
                            self.log_text_edit.append(err)
                        else:
                            saved_files += 1
                    else:
                        self.log_text_edit.append(f"копирование \"{filename}\" в \"{dest_filename}\": конечный файл уже существует")
            
            self.progress_bar.setValue(index + 1)
        
        self.pause_button.hide()
        self.status_label.setText(f"Сканирование папки \"{dir_to_scan}\" завершено. Проверено файлов: {len(all_files)}, обнаружено искомых файлов: {found_files}, скопировано файлов: {saved_files}.")

class StartWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Поиск файлов по хэшу")
        self.setFixedWidth(800)
        rect = self.frameGeometry()
        rect.moveCenter(QDesktopWidget().availableGeometry().center())
        self.move(rect.topLeft())

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        vbox_layout = QVBoxLayout()
        central_widget.setLayout(vbox_layout)
        
        form_layout = QFormLayout()

        self.dir_to_scan = QLineEdit()
        self.dir_to_scan.setReadOnly(True)
        form_layout.addRow("Какую папку сканировать", self.dir_to_scan)
        dir_to_scan_select_button = QPushButton("выбрать папку")
        dir_to_scan_select_button.clicked.connect(self.select_dir_to_scan)
        form_layout.addRow("", dir_to_scan_select_button)

        self.dir_for_results = QLineEdit()
        self.dir_for_results.setReadOnly(True)
        form_layout.addRow("Куда копировать найденные файлы", self.dir_for_results)
        dir_to_scan_select_button = QPushButton("выбрать папку")
        dir_to_scan_select_button.clicked.connect(self.select_dir_for_results)
        form_layout.addRow("", dir_to_scan_select_button)
  
        label1 = QLabel()
        label1.setText("(если не выбирать папку для копирования, то будет выполнен только поиск файлов)")
        label1.setWordWrap(True);
        form_layout.addRow("", label1)
  
        run_button = QPushButton("Поехали!")
        run_button.clicked.connect(self.run)
        
        vbox_layout.addLayout(form_layout)
        vbox_layout.addWidget(run_button)
        
    def select_dir_to_scan(self):
        dir = QFileDialog.getExistingDirectory(None, "Выберите папку для сканирования")
        if dir:
            self.dir_to_scan.setText(dir)

    def select_dir_for_results(self):
        dir = QFileDialog.getExistingDirectory(None, "Выберите папку для копирования найденных файлов")
        if dir:
            self.dir_for_results.setText(dir)
            
    def run(self):
        if self.dir_to_scan.text() == '':
            QErrorMessage(parent=self).showMessage('Папка для сканирования не выбрана!')
            return
        if self.dir_to_scan.text() == self.dir_for_results.text():
            QErrorMessage(parent=self).showMessage('Выбранные папки не должны совпадать!')
            return
        self.close()
        self.main_window = MainWindow()
        self.main_window.showMaximized()
        self.main_window.search_files(os.path.normpath(self.dir_to_scan.text()), os.path.normpath(self.dir_for_results.text()))

if __name__ == '__main__':
    app = QApplication([])
    window = StartWindow()
    window.show()
    sys.exit(app.exec_())
