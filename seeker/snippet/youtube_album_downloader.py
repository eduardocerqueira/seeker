#date: 2025-11-05T17:00:56Z
#url: https://api.github.com/gists/f008b295f87c0c26348494662c8b39b4
#owner: https://api.github.com/users/all3f0r1

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
YouTube Album Downloader
Interface graphique pour télécharger des albums YouTube par chapitres en MP3
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import subprocess
import threading
import os
import re
from pathlib import Path

class YouTubeAlbumDownloader:
    def __init__(self, root):
        self.root = root
        self.root.title("YouTube Album Downloader")
        self.root.geometry("700x550")
        self.root.resizable(True, True)
        
        # Définir le répertoire de téléchargement par défaut
        self.music_dir = Path.home() / "Musique"
        if not self.music_dir.exists():
            # Essayer "Music" en anglais si "Musique" n'existe pas
            self.music_dir = Path.home() / "Music"
            if not self.music_dir.exists():
                # Créer le dossier Musique si aucun n'existe
                self.music_dir = Path.home() / "Musique"
                self.music_dir.mkdir(exist_ok=True)
        
        self.is_downloading = False
        self.setup_ui()
        
    def setup_ui(self):
        """Configurer l'interface utilisateur"""
        
        # Frame principal avec padding
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configurer le redimensionnement
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(4, weight=1)
        
        # Titre
        title_label = ttk.Label(main_frame, text="YouTube Album Downloader", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 15))
        
        # URL input
        url_label = ttk.Label(main_frame, text="URL YouTube:")
        url_label.grid(row=1, column=0, sticky=tk.W, pady=5)
        
        self.url_entry = ttk.Entry(main_frame, width=60)
        self.url_entry.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        self.url_entry.bind('<Control-v>', self.on_paste)
        self.url_entry.bind('<Command-v>', self.on_paste)
        
        # Répertoire de destination
        dest_label = ttk.Label(main_frame, text=f"Destination: {self.music_dir}")
        dest_label.grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=5)
        
        # Zone de log
        log_label = ttk.Label(main_frame, text="Journal de téléchargement:")
        log_label.grid(row=4, column=0, sticky=(tk.W, tk.N), pady=(10, 5))
        
        self.log_text = scrolledtext.ScrolledText(main_frame, height=15, width=70, 
                                                  state='disabled', wrap=tk.WORD)
        self.log_text.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        # Barre de progression
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=6, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        # Boutons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=7, column=0, columnspan=2, pady=10)
        
        self.download_button = ttk.Button(button_frame, text="Télécharger", 
                                         command=self.start_download, width=20)
        self.download_button.pack(side=tk.LEFT, padx=5)
        
        self.clear_button = ttk.Button(button_frame, text="Effacer le journal", 
                                      command=self.clear_log, width=20)
        self.clear_button.pack(side=tk.LEFT, padx=5)
        
        # Instructions
        info_text = ("Instructions: Collez l'URL d'une vidéo YouTube contenant des chapitres.\n"
                    "L'album sera téléchargé en MP3 avec chaque chapitre séparé.")
        info_label = ttk.Label(main_frame, text=info_text, foreground="gray", 
                              font=("Arial", 9), wraplength=650)
        info_label.grid(row=8, column=0, columnspan=2, pady=(5, 0))
        
    def on_paste(self, event=None):
        """Gérer le collage d'URL"""
        # Laisser l'événement se propager normalement
        return None
        
    def log_message(self, message):
        """Ajouter un message au journal"""
        self.log_text.config(state='normal')
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state='disabled')
        
    def clear_log(self):
        """Effacer le journal"""
        self.log_text.config(state='normal')
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state='disabled')
        
    def get_video_title(self, url):
        """Obtenir le titre de la vidéo pour créer le dossier"""
        try:
            result = subprocess.run(
                ['yt-dlp', '--get-title', url],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                title = result.stdout.strip()
                # Nettoyer le titre pour en faire un nom de dossier valide
                title = re.sub(r'[<>:"/\\|?*]', '', title)
                title = title.strip()
                return title
            return None
        except Exception as e:
            self.log_message(f"Erreur lors de la récupération du titre: {str(e)}")
            return None
            
    def download_album(self, url):
        """Télécharger l'album YouTube"""
        try:
            self.log_message("=" * 60)
            self.log_message("Début du téléchargement...")
            self.log_message(f"URL: {url}")
            
            # Obtenir le titre de la vidéo
            self.log_message("Récupération des informations de la vidéo...")
            video_title = self.get_video_title(url)
            
            if video_title:
                self.log_message(f"Titre: {video_title}")
                output_dir = self.music_dir / video_title
                output_dir.mkdir(exist_ok=True)
                self.log_message(f"Dossier de destination: {output_dir}")
            else:
                output_dir = self.music_dir
                self.log_message("Impossible de récupérer le titre, téléchargement dans le dossier Musique")
            
            # Construire la commande yt-dlp
            output_template = str(output_dir / "%(section_number)s - %(section_title)s.%(ext)s")
            
            cmd = [
                'yt-dlp',
                url,
                '--extract-audio',
                '--audio-format', 'mp3',
                '--split-chapters',
                '--force-keyframes-at-cuts',
                '-o', output_template,
                '--progress',
                '--newline'
            ]
            
            self.log_message("\nTéléchargement en cours...")
            self.log_message("-" * 60)
            
            # Exécuter la commande
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Lire la sortie en temps réel
            for line in process.stdout:
                line = line.strip()
                if line:
                    self.log_message(line)
            
            process.wait()
            
            if process.returncode == 0:
                self.log_message("-" * 60)
                self.log_message("✓ Téléchargement terminé avec succès!")
                self.log_message(f"Fichiers sauvegardés dans: {output_dir}")
                self.log_message("=" * 60)
                messagebox.showinfo("Succès", 
                                   f"Album téléchargé avec succès!\n\nEmplacement:\n{output_dir}")
            else:
                self.log_message("-" * 60)
                self.log_message("✗ Erreur lors du téléchargement")
                self.log_message("=" * 60)
                messagebox.showerror("Erreur", 
                                    "Le téléchargement a échoué. Consultez le journal pour plus de détails.")
                
        except Exception as e:
            self.log_message(f"\n✗ Exception: {str(e)}")
            self.log_message("=" * 60)
            messagebox.showerror("Erreur", f"Une erreur s'est produite:\n{str(e)}")
        finally:
            self.is_downloading = False
            self.progress.stop()
            self.download_button.config(state='normal')
            self.url_entry.config(state='normal')
            
    def start_download(self):
        """Démarrer le téléchargement dans un thread séparé"""
        url = self.url_entry.get().strip()
        
        if not url:
            messagebox.showwarning("Attention", "Veuillez entrer une URL YouTube")
            return
            
        if not url.startswith(('http://', 'https://')):
            messagebox.showwarning("Attention", "L'URL doit commencer par http:// ou https://")
            return
            
        if self.is_downloading:
            messagebox.showinfo("Information", "Un téléchargement est déjà en cours")
            return
            
        self.is_downloading = True
        self.download_button.config(state='disabled')
        self.url_entry.config(state='disabled')
        self.progress.start()
        
        # Lancer le téléchargement dans un thread séparé
        download_thread = threading.Thread(target=self.download_album, args=(url,), daemon=True)
        download_thread.start()

def main():
    root = tk.Tk()
    app = YouTubeAlbumDownloader(root)
    root.mainloop()

if __name__ == "__main__":
    main()
