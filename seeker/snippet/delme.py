#date: 2026-01-02T17:07:44Z
#url: https://api.github.com/gists/0443df50863a368b3fa973593bf85f64
#owner: https://api.github.com/users/zanodor

import os
import re
import logging
import tkinter as tk
from tkinter import filedialog, ttk, scrolledtext, messagebox
from datetime import datetime
from pathlib import Path
import subprocess
import shutil
import urllib.parse
import json
import random
import string
from collections import defaultdict
import platform

"""
=== NAMING GUIDE ===

1. **actress_name**: Bare name extracted from filename before nationality
   - Example: "Alexandra" or "Alexandra AKA Anna T"
   - Extracted in: parse_filename_metadata()
   - Used for: Basic identification

2. **actress_aliases**: List of AKA names
   - Example: ["Anna T"]
   - Extracted in: parse_filename_metadata() when " AKA " found
   - Used for: Alternative names in frontmatter aliases

3. **actress_uuid**: Unique identifier (3 letters + 5 alphanumeric)
   - Example: "AlxA1B2C"
   - Generated in: generate_actress_uuid() or found via find_existing_actress_uuid()
   - Used for: Filename, folder structure, linking

4. **title_with_descriptors**: Full display name with nationality and descriptors
   - Example: "Alexandra Australian teen"
   - Built in: create_actress_note()
   - Used for: YAML title field, folder name, markdown heading, aliases
   - Rules: actress_name + nationality + common_descriptors (up to 5, stops at terminal)

5. **full_alias**: Copy of title_with_descriptors
   - Used for: Creating actress folder name via create_actress_folder_and_move()ac
   - Example: "AlxA1B2C_001_OMG.mp4"
   - Generated in: generate_short_filename()
   - Format: {actress_uuid}_{sequence:03d}_{movie_code}_{rating}.ext
"""

# OS detection
IS_WINDOWS = platform.system() == 'Windows'

# Add color support for Windows
if IS_WINDOWS:
    import ctypes
    kernel32 = ctypes.windll.kernel32
    kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)

# ANSI color codes
class Colors:
    RED = '\033[91m'
    YELLOW = '\033[93m'
    GREEN = '\033[92m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

# Configuration
NATIONALITIES = {
    'MAGYAR': 'Hungarian', 'Hungarian': 'Hungarian', 'Slovakian': 'Slovakian',
    'MAGYAR-Slovakian': 'Hungarian', 'MAGYAR-Romanian': 'Hungarian', 'MAGYAR-Serbian': 'Hungarian',
    'MAGYAR-Indian': 'Hungarian', 'Polish-German': 'Polish', 'Czech-German': 'Czech',
    'Polish-French': 'Polish', 'Algerian-French': 'Algerian', 'Moroccan-French': 'Moroccan',
    'MAGYAR-Ukrainian': 'Hungarian-Ukrainian', 'Czech': 'Czech', 'German': 'German',
    'Austrian': 'Austrian', 'Swiss': 'Swiss', 'Italian': 'Italian', 'Spanish': 'Spanish',
    'Portuguese': 'Portuguese', 'French': 'French', 'English': 'British', 'UK': 'British',
    'Welsh': 'Welsh', 'Irish': 'Irish', 'Dutch': 'Dutch', 'Belgian': 'Belgian',
    'Danish': 'Danish', 'Swedish': 'Swedish', 'Norwegian': 'Norwegian', 'Finnish': 'Finnish',
    'Baltic': 'Baltic', 'Latvian': 'Latvian', 'Lithuanian': 'Lithuanian', 'Estonian': 'Estonian',
    'Slavic': 'Slavic', 'Moldavian': 'Moldavian', 'Polish': 'Polish', 'Russian': 'Russian',
    'Belorussian': 'Belorussian', 'Bulgarian': 'Bulgarian', 'Serbian': 'Serbian', 'Slovenian': 'Slovenian',
    'Ukrainian': 'Ukrainian', 'Romanian': 'Romanian', 'Greek': 'Greek',
    'Israeli': 'Israeli', 'Turkish': 'Turkish', 'Arabic': 'Arabic', 'Tunisian': 'Tunisian',
    'Algerian': 'Algerian', 'Moroccan': 'Moroccan', 'Japanese': 'Japanese',
    'Vietnamese': 'Vietnamese', 'Korean': 'Korean', 'Chinese': 'Chinese', 'Indian': 'Indian',
    'Desi': 'Desi', 'Bengali': 'Bengali', 'Pakistani': 'Pakistani', 'Sri Lankan': 'Sri Lankan',
    'Australian': 'Australian', 'New Zealander': 'New Zealander', 'Canadian': 'Canadian',
    'USA': 'American', 'American': 'American', 'Mexican': 'Mexican', 'Brazilian': 'Brazilian',
    'Argentinian': 'Argentinian', 'Chilean': 'Chilean', 'Colombian': 'Colombian', 'Paraguayan': 'Paraguayan', 
    'Cuban': 'Cuban', 'Salvadorean': 'Salvadorean', 'Salvadorian': 'Salvadorean', 'Costa Rican': 'Costa Rican',
    'Puerto Rican': 'Puerto Rican',
    'South African': 'South African'
}

TAG_KEYWORDS = [
    # Existing quality/rating tags
    'WOW', 'OMG', 'sensational', 'gorgeous', 'amazing', 'fantastic',
    
    # Facial features
    'big-nose', 'big-nosed', 'pointed-nose', 'hook-nose', 'crooked-nose', 'button-nose', 'ugly-nose',
    'strange-teeth', 'uneven-teeth', 'uneven-front-teeth', 'gaps-between-teeth', 'toothless',
    'large-teeth', 'large-chin', 'weak-chin', 'curved-nose',
    'cross-eyed', 'large-eyed', 'gorgeous-eyes', 'beautiful-eyes',
    'wide-face', 'pixie-face', 'duck-faced', 'flat-faced',
    'mole-on-face', 'mole', 'moles-on-chest', 'moles-on-body', 'birthmark', 'beauty-mark', 'beauty-marks',
    
    # Appearance quality
    'beautiful', 'pretty', 'gorgeous', 'stunning', 'attractive', 'sensual',
    'plain', 'plain-looking', 'average', 'average-pretty', 'ugly-pretty',
    'fearsome', 'foxy', 'sweet', 'elegant', 'soft-titted',
    'innocent', 'shy', 'perverted', 'whorish', 'skanky',
    
    # Skin quality
    'acned', 'pimpled', 'scarred', 'pimpled-ass', 'bad-facial-skin',
    'creamy-white-skinned', 'pale-skinned', 'white-skinned', 'nice-skin',
    'nice-white-skin', 'clear-skin',
    
    # Hair styles
    'hairy', 'hirsute', 'short-haired', 'long-haired', 'frizzy-haired',
    'pigtailed', 'pig-tailed', 'pigtails', 'short-ish-haired',
    'dyed-haired', 'dyed-hair', 'dyed-blond', 'dyed-blonde',
    'brunette', 'blonde', 'redhead',
    
    # Hairy body parts (keep specific)
    'hairy-cunny', 'hairy-pussy', 'hairy-asshole', 'hairy-assed',
    'hairy-armpits', 'hairy-legs', 'hairy-legged', 'hairy-ass',
    'full-bush', 'treasure-trail', 'slightly-hairy',
    
    # Breast descriptors
    'saggy-titted', 'huge-titted', 'big-titted', 'busty',
    'small-titted', 'flat-chested', 'nice-titted', 'huge-saggy',
    'soft-titted', 'huge-breasted', 'large-breasted', 'big-tits',
    'nice-tits', 'great-tits', 'fantastic-tits', 'torpedo-tits',
    'great-saggy-tits', 'fabulous-big-tits', 'huge-boobs',
    'fake-tits', 'pumped-tits', 'large-areolas', 'big-areolas',
    
    # Body types
    'skinny', 'thin', 'slim', 'slender', 'child-bodied', 'small-bodied',
    'small-framed', 'tiny', 'petite', 'chunky', 'curvy',
    'chubby', 'plumper', 'plump', 'fattier', 'chubbier',
    'fat', 'BBW', 'huge',
    'tall',
    
    # Body parts quality
    'big-assed', 'good-assed', 'nice-ass', 'great-ass', 'fantastic-ass',
    'gorgeous-ass', 'bubble-butt-ass', 'huge-ass', 'flabby-body',
    'nice-body', 'good-body', 'gorgeous-body', 'great-body', 'perfect-body',
    'nice-tummy', 'nice-belly', 'good-hips',
    'gorgeous-thighs', 'chunky-thighed', 'good-thighs',
    'meaty-pussy-lips', 'big-pussy-lips', 'huge-pussy-lips',
    
    # Age/maturity
    'teen', 'teenie', 'daughter', 'young', 'young-looking', 'younger',
    'mature', 'older', 'mom', 'mother', 'granny',
    'milf', 'wife', 'housewife',
    
    # Ethnicity/appearance
    'roma', 'gypsy', 'half-roma', 'mixed', 'mixed-race', 'mestiza', 'ebony', 'native',
    'latina', 'Indian-looking', 'Neanderthal',
    
    # Style/accessories
    'nerdy', 'glasses-wearing', 'glasses', 'with-glasses', 'without-glasses',
    'emo',
    
    # Body modifications
    'tattooed', 'tattoo', 'tattoos', 'with-tattoos',
    'dragon-tattoo', 'scorpion-tattoo', 'tiger-tattoo',
    'rabbit-tattoo', 'rose-tattoo', 'heart-tattoo',
    'heart-and-arrow-tattoo', 'cupid-tattoo', 'butterfly-tattoo',
    'tribal-tattoo', 'snake-tattoo', 'devil-tattoo', 'angel-tattoo',
    'cross-tattoo', 'playboy-tattoo', 'clown-tattoo', 'cherub-tattoo',
    'dolphin-tattoo', 'small-tattoos', 'small-heart-tattoo',
    'pierced', 'with-piercings',
    
    # Genital state
    'shaved', 'unshaved',
    
    # Personality/demeanor
    'next-door', 'cumwhore',
    
    # Production quality
    'amateur', 'homemade', 'home-made', 'semi-pro', 'pro-am',

    # Production genre
    'scat', 'piss', 'casting', 'midget', 'dwarf', 'handicapped',
    
    # Era
    'retro', 'vintage', 'classic', 'current', 'modern',
    
    # Special attributes
    'rare', 'kuriózum', 'deaf-mute', 'pregnant', 'preggo', 'post-natal',
    
    # Negative features
    'butterface', 'ugly',
    
    # Miscellaneous
    'natural', 'fake', 'enhanced',
    'babe', 'beauty', 'hottie', 'sweetie', 'cutie', 'chick',
    'swinger', 'all-time-fave', 'fave', 'Goddess',
    'negro-lover', 'BBC-lover',
]

# Descriptor categories for title building
DESCRIPTOR_SETS = {
    'body_types': {
        'skinny', 'thin', 'slim', 'slender', 
        'chunky', 'chubby', 'plumper', 'BBW', 'fat', 
        'average', 'curvy', 'busty', 'petite', 
        'tall', 'huge', 'tiny'
    },
    'hair_colors': {
        'blonde', 'brunette', 'redhead', 'blond'
    },
    'age_terms': {
        'teen', 'chick', 'daughter', 'mature', 'milf', 'granny', 
        'young', 'young-looking', 'mom', 'mother', 'mommy'
    },
    'quality_terms': {
        'gorgeous', 'beautiful', 'pretty', 
        'attractive', 'sensual', 'foxy', 
        'stunning', 'cute', 'cutie', 'sweetie',
        'babe', 'beauty', 'hottie', 'sensational',
        'amazing', 'fantastic'
    },
    'other_descriptors': {
        'hairy', 'hirsute', 'amateur', 'natural',
        'next-door', 'nerdy', 'shy', 'rare',
        'retro', 'vintage', 'classic', 'homemade',
        'butterface', 'glasses', 'pigtailed'
    }
}

# Terminal descriptors - these end the title
TERMINAL_DESCRIPTORS = {
    'BBW', 'MILF', 'PAWG', 'chubby', 'plumper', 'skinny', 'pretty', 'beauty',
    'ugly', 'sweetie', 'cutie', 'bunny', 'babe', 'teen', 'mature', 'granny', 'amateur',
    'wife', 'wifey', 'housewife', 'mom', 'mommy', 'mother', 'daughter', 
    'girl', 'lady', 'woman', 'Goddess', 'porn star', 'pornstar', 'chick', 'hottie',
    'sweetness', 'csaj', 'csajszi', 'slut', 'bitch', 'whore', 'pixie',
    'neighbor', 'fave',
    'blonde', 'redhead', 'brunette', 'dyed-blonde'
}

NATIONALITY_PATTERN = '|'.join(map(re.escape, NATIONALITIES.keys()))

class VideoProcessorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Unified Video Processor V2.5.2 - Smart UUID Naming")
        self.root.geometry("1000x750")
        
        self.single_video_file = tk.StringVar()
        self.video_folder = tk.StringVar()
        self.stash_base = tk.StringVar(value="C:/Users/ODOR/Documents/Obsidian/STASH")
        self.partners_file = tk.StringVar(value="C:/Users/ODOR/Documents/Obsidian/STASH/pages/Dummy File Populated with PartnersWith Property Values for Modal Forms Assisted Note Creation.md")
        self.dry_run = tk.BooleanVar(value=True)
        self.include_title_in_aliases = tk.BooleanVar(value=True)
        self.selected_video_files = []
        
        self.known_partners = set()
        self.actress_database = {}
        self.video_metadata = []
        self.rename_log = []
        self.used_uuids = set()
        self.actress_index = {}
        self.video_index = {}

        self.create_widgets()
        self.setup_logging()
    
    def setup_logging(self):
        log_dir = "C:/Users/ODOR/Documents/Obsidian/STASH/logs"
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, f'video_processing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        
        self.log_file_path = log_file
    
    def create_widgets(self):
        header = tk.Label(self.root, text="Unified Video Processor V2.5.2",
                         font=("Arial", 16, "bold"), pady=10, bg="#1976D2", fg="white")
        header.pack(fill="x")
        
        config_frame = tk.LabelFrame(self.root, text="Configuration", font=("Arial", 10, "bold"), pady=10)
        config_frame.pack(fill="x", padx=20, pady=10)
        
        tk.Label(config_frame, text="Single Video File to Process:", font=("Arial", 9, "bold")).pack(anchor="w", padx=10)
        file_frame = tk.Frame(config_frame)
        file_frame.pack(fill="x", padx=10, pady=5)
        tk.Entry(file_frame, textvariable=self.single_video_file, width=80).pack(side="left", padx=(0, 10))
        tk.Button(file_frame, text="Browse", command=self.browse_single_video_file, bg="#4CAF50", fg="white").pack(side="left")
        
        tk.Label(config_frame, text="Video Folder to Process:", font=("Arial", 9, "bold")).pack(anchor="w", padx=10)
        folder_frame = tk.Frame(config_frame)
        folder_frame.pack(fill="x", padx=10, pady=5)
        tk.Entry(folder_frame, textvariable=self.video_folder, width=80).pack(side="left", padx=(0, 10))
        tk.Button(folder_frame, text="Browse", command=self.browse_video_folder, bg="#4CAF50", fg="white").pack(side="left")
        
        tk.Label(config_frame, text="STASH Base Folder:", font=("Arial", 9, "bold")).pack(anchor="w", padx=10, pady=(10,0))
        tk.Entry(config_frame, textvariable=self.stash_base, width=80).pack(anchor="w", padx=10, pady=5)
        
        tk.Label(config_frame, text="Partners Dictionary File:", font=("Arial", 9, "bold")).pack(anchor="w", padx=10)
        partners_frame = tk.Frame(config_frame)
        partners_frame.pack(fill="x", padx=10, pady=5)
        tk.Entry(partners_frame, textvariable=self.partners_file, width=70).pack(side="left", padx=(0, 10))
        tk.Button(partners_frame, text="Browse", command=self.browse_partners_file, bg="#4CAF50", fg="white").pack(side="left")
        
        options_frame = tk.Frame(config_frame)
        options_frame.pack(fill="x", padx=10, pady=10)
        tk.Checkbutton(options_frame, text="📝 Include title in aliases (when actress has AKA aliases)", 
                      variable=self.include_title_in_aliases, font=("Arial", 9)).pack(anchor="w")
        tk.Checkbutton(options_frame, text="🔍 Dry Run (Preview only, doesn't rename. - Make copies of folders before going live.)", 
                      variable=self.dry_run, font=("Arial", 10, "bold"), fg="#FF6B00").pack(anchor="w")
        
        process_btn = tk.Button(self.root, text="🚀 Process Videos", command=self.process_videos,
                               font=("Arial", 12, "bold"), bg="#2196F3", fg="white", 
                               pady=15, cursor="hand2")
        process_btn.pack(pady=10)
        
        reprocess_btn = tk.Button(self.root, text="🔄 Reprocess Thumbnails", command=self.reprocess_thumbnails,
                               font=("Arial", 11, "bold"), bg="#9C27B0", fg="white",
                               pady=10, cursor="hand2")
        reprocess_btn.pack(pady=5)
        
        self.progress = ttk.Progressbar(self.root, mode='indeterminate')
        self.progress.pack(fill="x", padx=20, pady=5)
        
        results_frame = tk.LabelFrame(self.root, text="Processing Log", font=("Arial", 10, "bold"))
        results_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        self.log_text = scrolledtext.ScrolledText(results_frame, wrap=tk.WORD, font=("Consolas", 9))
        self.log_text.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.log_text.tag_config("error", foreground="#D32F2F", font=("Consolas", 9, "bold"))
        self.log_text.tag_config("warning", foreground="#FF6B00", font=("Consolas", 9, "bold"))
        self.log_text.tag_config("success", foreground="#388E3C", font=("Consolas", 9, "bold"))
        self.log_text.tag_config("info", foreground="#1976D2")
        
        self.status_var = tk.StringVar(value="Ready")
        status_bar = tk.Label(self.root, textvariable=self.status_var, 
                            relief=tk.SUNKEN, anchor="w", bg="#E0E0E0")
        status_bar.pack(fill="x", side="bottom")
    
    def browse_video_folder(self):
        # Try to use multiple folder selection if available
        try:
            import tkinter.filedialog as fd
            from tkinter import Tk
            
            # Create a hidden root window for the dialog
            temp_root = Tk()
            temp_root.withdraw()
            
            # Ask user if they want to select multiple folders
            response = messagebox.askyesno(
                "Folder Selection",
                "Do you want to select multiple folders?\n\n"
                "Yes = Select multiple folders\n"
                "No = Select single folder"
            )
            
            if response:
                # Multiple folder selection
                folders = []
                messagebox.showinfo(
                    "Multiple Folder Selection",
                    "Select folders one by one.\n"
                    "Click 'Cancel' when done selecting all folders."
                )
                
                while True:
                    folder = filedialog.askdirectory(title="Select Video Folder (Cancel when done)")
                    if not folder:
                        break
                    folders.append(folder)
                    
                    if not messagebox.askyesno("Continue?", f"Added: {folder}\n\nSelect another folder?"):
                        break
                
                if folders:
                    self.video_folder.set('; '.join(folders))
                    self.log(f"Selected {len(folders)} folders", "success")
            else:
                # Single folder selection
                folder = filedialog.askdirectory(title="Select Video Folder to Process")
                if folder:
                    self.video_folder.set(folder)
            
            temp_root.destroy()
            
        except Exception as e:
            # Fallback to single folder selection
            folder = filedialog.askdirectory(title="Select Video Folder to Process")
            if folder:
                self.video_folder.set(folder)
    
    def browse_single_video_file(self):
        files = filedialog.askopenfilenames(
            title="Select Video File(s)",
            filetypes=[("Video files", "*.mp4 *.avi *.mkv *.mov *.wmv *.flv *.mpg *.mpeg *.m4v *.webm *.3gp *.ogv *.asf *.asx *.vob *.ts *.m2ts *.mts *.mxf *.divx *.f4v *.rm *.rmvb *.dv"), ("All files", "*.*")]
        )
        if files:
            self.selected_video_files = list(files)
            self.single_video_file.set('; '.join(files))
    
    def browse_partners_file(self):
        file = filedialog.askopenfilename(
            title="Select Partners Dictionary File",
            filetypes=[("Markdown files", "*.md"), ("All files", "*.*")]
        )
        if file:
            self.partners_file.set(file)
    
    def log(self, message, tag="info"):
        self.log_text.insert(tk.END, message + "\n", tag)
        self.log_text.see(tk.END)
        self.root.update()
        
        if tag == "error":
            logging.error(message)
        elif tag == "warning":
            logging.warning(message)
        else:
            logging.info(message)
    
    def load_partners(self):
        partners_file = self.partners_file.get()
        
        if not os.path.exists(partners_file):
            self.log(f"⚠️ Partners file not found: {partners_file}", "warning")
            return
        
        try:
            with open(partners_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.startswith('  - '):
                        partner = line[4:].strip()
                        if partner:
                            self.known_partners.add(partner)
            
            self.log(f"✅ Loaded {len(self.known_partners)} known partners", "success")
        except Exception as e:
            self.log(f"❌ Error loading partners: {str(e)}", "error")
    
    def load_actress_index(self, stash_base):
        self.actress_index = {'aliases': [], 'files': {}}
        for ch in map(chr, range(65, 91)):
            dir_path = os.path.join(stash_base, ch)
            if not os.path.isdir(dir_path):
                continue
            for name in os.listdir(dir_path):
                if name.lower().endswith('.md'):
                    file_path = os.path.join(dir_path, name)
                    old_data, content = self.read_existing_markdown(file_path)
                    uuid = os.path.splitext(name)[0]
                    aliases = []
                    if old_data and 'aliases' in old_data:
                        aliases.extend(old_data['aliases'])
                    if content:
                        m = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
                        if m:
                            aliases.append(m.group(1).strip())
                    for a in aliases:
                        self.actress_index['aliases'].append((a.lower(), uuid))
                    self.actress_index['files'][uuid] = old_data or {}
    
    def load_video_index(self, stash_base):
        self.video_index = {'aliases': []}
        videos_root = os.path.join(stash_base, 'Videos')
        if not os.path.isdir(videos_root):
            return
        for ch in map(chr, range(65, 91)):
            dir_path = os.path.join(videos_root, ch)
            if not os.path.isdir(dir_path):
                continue
            for name in os.listdir(dir_path):
                if name.lower().endswith('.md'):
                    file_path = os.path.join(dir_path, name)
                    old_data, _ = self.read_existing_markdown(file_path)
                    if not old_data:
                        continue
                    aliases = old_data.get('aliases', [])
                    actress_field = old_data.get('actress', '')
                    m = re.search(r'\[\[(.*?)\]\]', actress_field)
                    uuid = m.group(1) if m else ''
                    if uuid:
                        for a in aliases:
                            self.video_index['aliases'].append((a.lower(), uuid))

    def get_common_descriptors(self, videos):
        if not videos:
            return []
        terminal_lower = {d.lower() for d in TERMINAL_DESCRIPTORS}
        if not videos[0].get('descriptors'):
            return []
        common = {d.lower() for d in videos[0]['descriptors']}
        for v in videos[1:]:
            common &= {d.lower() for d in v.get('descriptors', [])}
            if not common:
                break
        common_terminals = [d for d in videos[0]['descriptors'] if d.lower() in common and d.lower() in terminal_lower]
        return common_terminals[:1]

    def get_title_noun_descriptor(self, videos):
        if not videos:
            return ''
        order = videos[0].get('descriptors', [])
        if not order:
            return ''
        counts = {}
        for v in videos:
            for d in v.get('descriptors', []):
                dl = d.lower()
                counts[dl] = counts.get(dl, 0) + 1
        n = len(videos)
        threshold = max(1, (n + 1) // 2)
        candidates = {
            'mom','mother','mommy','milf','granny',
            'bbw','pawg','chubby','plumper','skinny',
            'wife','wifey','housewife','girl','lady','woman','pornstar','porn star',
            'babe','beauty','hottie','cutie','chick','pixie','teen','daughter','fave'
        }
        normalization = {'blond': 'blonde', 'wifey': 'wife', 'porn star': 'pornstar'}
        # Prefer terminal noun over hair color if present
        hair_terms = ['blonde','brunette','redhead','blond']
        hair_counts = {t: counts.get(t, 0) for t in hair_terms}
        best_hair = max(hair_terms, key=lambda t: hair_counts[t])
        priority = [
            'mom','mother','mommy','milf','granny',
            'bbw','pawg','chubby','plumper','skinny',
            'wife','wifey','housewife','girl','woman','lady','pornstar','porn star',
            'babe','beauty','hottie','cutie','chick',
            'teen','daughter','pixie',
            'fave'
        ]
        for term in priority:
            c = counts.get(term, 0)
            if term in candidates and c >= threshold:
                orig = next((x for x in order if x.lower() == term), term)
                norm = normalization.get(orig.lower(), orig)
                return norm
        # If no noun present, fall back to hair color by MODE
        if hair_counts[best_hair] > 0:
            return normalization.get(best_hair, best_hair)
        return ''

    def prefix_from_filename(self, filename, n=3):
        clean = filename.replace('(trimmed)', '').strip()
        words = re.findall(r'[A-Za-zÁÉÍÓÖŐÚÜŰáéíóöőúüű-]+', clean)
        return ' '.join(words[:n]).lower()
    
    def find_existing_actress_uuid(self, filename, stash_base):
        if not self.video_index:
            self.load_video_index(stash_base)
        prefix = self.prefix_from_filename(filename)
        for alias, uuid in self.video_index['aliases']:
            if alias.startswith(prefix) or prefix in alias:
                return uuid
        if not self.actress_index:
            self.load_actress_index(stash_base)
        for alias, uuid in self.actress_index['aliases']:
            if alias.startswith(prefix) or prefix in alias:
                return uuid
        return None
    
    def generate_actress_uuid(self, actress_name):
        letters = ''.join(c for c in actress_name if c.isalpha())[:3]
        
        while True:
            random_part = ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))
            uuid = letters + random_part
            
            if uuid not in self.used_uuids:
                self.used_uuids.add(uuid)
                return uuid
    
    def parse_filename_metadata(self, filename):
        metadata = {
            'original_filename': filename,
            'actress_name': '',
            'actress_aliases': [],
            'nationality': '',
            'descriptors': [],
            'tags': [],
            'partners': [],
            'movie': '',
            'movie_alt': [],
            'rating': '',
            'rating_stars': '',
            'has_bbc': 'BBC' in filename,
            'has_anim': 'ANIM' in filename,
            'is_trimmed': filename.endswith('(trimmed)') or '(trimmed).' in filename,
            'timestamps': {},
            'bodytype': '',
            'haircolor': '',
            'actressactive_when': '',
            'madeupname': False,
            'version': '',
            'aie': False
        }
        
        clean_filename = filename
        if clean_filename.startswith('ANIM '):
            clean_filename = clean_filename[5:]
        
        clean_filename = clean_filename.replace('(trimmed)', '').strip()
        def strip_caps_tail(s):
            allowed = ('OMG','WOW','HQ','LQ','MQ','HHQ','VHQ','AIE')
            m = re.search(r'(?:\s+([A-ZÁÉÍÓÖŐÚÜŰ]{2,}(?:\s+[A-ZÁÉÍÓÖŐÚÜŰ]{2,})+))\s*$', s)
            if m:
                phrase = m.group(1)
                if not any(tok in phrase for tok in allowed):
                    return s[:m.start()].strip()
            return s
        clean_filename = strip_caps_tail(clean_filename)
        
        # Capture age prefix and add as tag (e.g., "14 year old ", "19 years old ")
        age_match = re.match(r'^(\d{2})\s+years?\s+old\s+', clean_filename, flags=re.IGNORECASE)
        if age_match:
            age = int(age_match.group(1))
            # Remove the age prefix from filename for parsing
            clean_filename = re.sub(r'^\d{2}\s+years?\s+old\s+', '', clean_filename, flags=re.IGNORECASE)
            # Add '_underage-looking' tag for 11-18, 'young' for 19-25
            # Underage looking actresses may need to be deleted from PC for safety!!! Good that Obsidian exists for helping with that!
            if 11 <= age <= 18:
                metadata['tags'].append('_underage-looking')
            elif 19 <= age <= 25:
                metadata['tags'].append('young')
        
        # First check filename for rating
        if ' OMG' in filename:
            metadata['rating'] = 'OMG'
            metadata['rating_stars'] = '⭐⭐⭐⭐⭐'
        elif ' WOW' in filename:
            metadata['rating'] = 'WOW'
            metadata['rating_stars'] = '⭐⭐⭐⭐'
        elif ' hmmm ' in filename or ' hmmmm ' in filename:
            metadata['rating'] = 'hmmm'
            metadata['rating_stars'] = '⭐⭐⭐⭐'
        elif ' hmm ' in filename:
            metadata['rating'] = 'hmm'
            metadata['rating_stars'] = '⭐⭐'
        
        # Find where nationality appears to stop name extraction
        nat_position = None
        found_nat_key = None
        for nat_key in NATIONALITIES.keys():
            pattern = fr'\b{re.escape(nat_key)}\b'
            match = re.search(pattern, clean_filename)
            if match:
                nat_position = match.start()
                found_nat_key = nat_key
                break
        
        # Extract name: everything before nationality, or before descriptors if no nationality
        if nat_position is not None:
            # We have nationality - name is everything before it
            full_name = clean_filename[:nat_position].strip()
        else:
            # No nationality - take first few capitalized words (fallback)
            name_match = re.match(r'^([A-Z][a-zA-ZáéíóöőúüűÁÉÍÓÖŐÚÜŰ-]+(?:\s+[A-Z][a-zA-ZáéíóöőúüűÁÉÍÓÖŐÚÜŰ-]+){0,2})', clean_filename)
            full_name = name_match.group(1).strip() if name_match else ""
        
        if full_name:
            if ' AKA ' in full_name:
                parts = full_name.split(' AKA ')
                metadata['actress_name'] = parts[0].strip()
                metadata['actress_aliases'] = [p.strip() for p in parts[1:]]
            else:
                metadata['actress_name'] = full_name.strip()
        
        nationality_match = re.search(fr'\b({NATIONALITY_PATTERN})\b', clean_filename)
        if nationality_match:
            metadata['nationality'] = NATIONALITIES[nationality_match.group(1)]
        
        # Extract version info first (all caps at end before extension)
        version_match = re.search(r'\s+([A-Z\s]+VERSION\d*)\s+(?=WOW|OMG|hmm+|\(trimmed\)|\(AIE\)|\.[a-z]+$)', filename)
        if version_match:
            metadata['version'] = version_match.group(1).strip()
            # Remove version from filename for movie parsing
            clean_filename_for_movie = filename[:version_match.start()]
        else:
            clean_filename_for_movie = filename
        clean_filename_for_movie = strip_caps_tail(clean_filename_for_movie)

        # Check for (AIE) flag
        if re.search(r'\(AIE\)', filename, re.IGNORECASE):
            metadata['aie'] = True
        
        # Extract all bracket info for Additional Info section (exclude years and AIE)
        # Extract all bracket info for Additional Info section (exclude years, AIE, and name disambiguation)
        bracket_info = []
        
        # First, find where the actress name ends (at nationality or first descriptor)
        name_end_pos = len(full_name) if full_name else 0
        
        for match in re.finditer(r'\(([^)]+)\)', filename):
            content = match.group(1).strip()
            bracket_pos = match.start()
            
            # Skip brackets that are part of the actress name (before nationality)
            if bracket_pos < name_end_pos:
                continue
            
            # Skip if it's just a 4-digit year
            if re.match(r'^\d{4}$', content):
                continue
            
            # Skip if it's just a digit (name disambiguation like (3))
            if re.match(r'^\d+$', content):
                continue
            
            # Skip if it's AIE
            if content.upper() == 'AIE':
                continue
            
            # Skip if it's trimmed
            if content.lower() == 'trimmed':
                continue
            
            # Keep everything else
            bracket_info.append(content)
        
        metadata['bracket_info'] = bracket_info
        
        # Parse movie title: find all "in [something]" and take the LAST valid one
        # Scene type indicators that are NOT movie titles (start with capitals but aren't movies)
        scene_types = {'FMM', 'FFM', 'FFMM', 'MFF', 'MMF', 'MMFF', 'DP', 'DPP', 'DAP', 'DVP'}
        
        # Find ALL "in [something]" patterns - capture until next "in" or end markers
        all_in_matches = list(re.finditer(r'\bin\s+(.+?)(?=\s+in\s+|\s+(?:WOW|OMG|hmm+)|\s+\(trimmed\)|\s+\(AIE\)|\s*$)', clean_filename_for_movie, re.IGNORECASE))
        
        movie_title = ""
        valid_candidates = []
        
        if all_in_matches:
            # Collect all valid candidates (skip lowercase starters and scene types)
            for match in all_in_matches:
                candidate = match.group(1).strip()
                
                # Remove any trailing parentheses content
                candidate = re.sub(r'\s*\([^)]*\)\s*$', '', candidate).strip()
                
                if not candidate:
                    continue
                
                # Get first word/character
                first_word = candidate.split()[0] if candidate else ""
                first_char = candidate[0]
                
                # Skip if it's a scene type indicator
                if first_word.upper() in scene_types:
                    continue
                
                # Skip if starts with lowercase English letter (locations like "kitchen", "bed", "barn")
                # But allow accented capitals like Á, É, etc. and digits
                if first_char.islower() and first_char.isascii():
                    continue
                
                # Valid if starts with: uppercase (English or accented), digit, or non-ASCII uppercase
                if first_char.isupper() or first_char.isdigit() or not first_char.isascii():
                    valid_candidates.append(candidate)
            
            # Take the LAST valid candidate (actual movie title, not intermediate locations)
            if valid_candidates:
                movie_title = valid_candidates[-1]
        
        if movie_title:
            # Remove ALL bracketed content from movie title
            movie_title = re.sub(r'\s*\([^)]*\)', '', movie_title).strip()
            
            # Remove any version strings that might still be attached
            movie_title = re.sub(r'\s+[A-Z\s]{2,}(?:VERSION|VERS|VER|V\d+)\s*$', '', movie_title).strip()
            
            # Remove quality tags (LQ, MQ, HQ, VHQ)
            movie_title = re.sub(r'\s+(?:LQ|MQ|HQ|HHQ|VHQ)\s*$', '', movie_title, flags=re.IGNORECASE).strip()
            
            # Split by ' or ' to get all movie titles (use word boundaries)
            if re.search(r'\bor\b', movie_title, re.IGNORECASE):
                movie_parts = [p.strip() for p in re.split(r'\s+or\s+', movie_title, flags=re.IGNORECASE)]
                # Clean each part of any trailing version strings
                movie_parts = [re.sub(r'\s+[A-Z\s]{2,}(?:VERSION|VERS|VER|V\d+)\s*$', '', p).strip() for p in movie_parts]
                metadata['movie'] = movie_parts[0]
                # All remaining parts are alt titles (as a list)
                metadata['movie_alt'] = [p for p in movie_parts[1:] if p]  # Filter empty strings
            else:
                metadata['movie'] = movie_title
                metadata['movie_alt'] = []
        
        extracted_tags = self.extract_tags(clean_filename, metadata['actress_name'])
        # Merge age-based tags with extracted tags (age tags are already in metadata['tags'])
        metadata['tags'] = list(set(metadata['tags'] + extracted_tags))
        metadata['partners'] = self.extract_partners(filename)
        
        # Extract descriptors more carefully - stop at action verbs
        if metadata['actress_name']:
            # Calculate where the FULL actress name section ends (including AKA aliases)
            full_name_length = len(full_name)
            remaining = clean_filename[full_name_length:].strip()
            
            # Remove nationality from remaining text BEFORE processing descriptors
            if metadata['nationality']:
                # Find and remove the FIRST occurrence of any matching nationality keyword
                for nat_key in NATIONALITIES.keys():
                    if NATIONALITIES[nat_key] == metadata['nationality']:
                        # Use word boundaries and remove exactly once
                        pattern = fr'\b{re.escape(nat_key)}\b\s*'
                        remaining = re.sub(pattern, '', remaining, count=1, flags=re.IGNORECASE).strip()
                        break  # Only remove the first match
            
            # Now extract descriptors from what's left
            desc_text = remaining
            
            # DEBUG
            self.log(f"    DEBUG parse: remaining after nationality = '{remaining}'")
            
            # FIRST: Split at action verbs to isolate actress descriptors from partner/scene descriptions
            action_splits = [
                r'\s+(?:gets?|sucks?|sucks? off|fucks?|fucked|does|is|shows?|performs?|had to try|plays?|dances?|interviews?|strips?|takes?|gives?)\s+',
                r'\s+(?:by|from|with|and)\s+(?!her|his|their|the|a|an)\s*',
                r'\'s\s+',
            ]
            
            for split_pattern in action_splits:
                parts = re.split(split_pattern, desc_text, flags=re.IGNORECASE, maxsplit=1)
                if len(parts) > 1:
                    desc_text = parts[0].strip()
                    self.log(f"    DEBUG parse: split at action verb/preposition, desc_text = '{desc_text}'")
                    break
            
            # SECOND: Stop at ' in ' if movie found (movie title marks end of actress description)
            if metadata['movie'] and ' in ' in desc_text:
                desc_text = desc_text.split(' in ')[0].strip()
                self.log(f"    DEBUG parse: stopped at ' in ', desc_text = '{desc_text}'")
            
            # Clean up trailing conjunctions and prepositions
            desc_text = re.sub(r'\s+(?:and|or|with|by|from)$', '', desc_text, flags=re.IGNORECASE).strip()
            
            self.log(f"    DEBUG parse: final desc_text = '{desc_text}'")
            
            # Split into words and filter
            words = desc_text.split()
            valid_descriptors = []
            
            stop_words = {'and', 'or', 'with', 'by', 'from', 'the', 'a', 'an', 'in', 'on', 'at', 'to', 'of', 'for'}
            
            # Combine all descriptor sets
            all_descriptors = set()
            for descriptor_set in DESCRIPTOR_SETS.values():
                all_descriptors.update(descriptor_set)
            
            seen = set()
            for word in words:
                word_lower = word.lower()
                if word_lower in stop_words or word_lower in seen:
                    continue
                if word_lower in all_descriptors:
                    valid_descriptors.append(word)
                    seen.add(word_lower)
            
            metadata['descriptors'] = valid_descriptors
        
        body_types = {
            'skinny': 'skinny', 'thin': 'skinny', 
            'slim': 'slim', 'slender': 'slim',
            'average': 'average',
            'chubby': 'chubby', 'plump': 'chubby',
            'BBW': 'BBW', 'fat': 'fat'
        }
        for key, value in body_types.items():
            if re.search(r'\b' + key + r'\b', clean_filename, re.IGNORECASE):
                metadata['bodytype'] = value
                break
        
        hair_colors = {
            'blonde': 'blonde', 'blond': 'blonde',
            'brunette': 'brown', 'brown-haired': 'brown',
            'redhead': 'red', 'ginger': 'red',
            'black-haired': 'black'
        }
        for color_term, color_value in hair_colors.items():
            if re.search(fr'\b{color_term}\b', clean_filename, re.IGNORECASE):
                metadata['haircolor'] = color_value
                break
        
        years = [int(m.group(1)) for m in re.finditer(r'\((\d{4})\)', filename) 
                 if 1950 <= int(m.group(1)) <= 2024]
        if years:
            avg_year = sum(years) / len(years)
            if 1950 <= avg_year < 1985:
                metadata['actressactive_when'] = 'retro'
            elif 1985 <= avg_year < 1997:
                metadata['actressactive_when'] = 'classic'
            elif 1997 <= avg_year < 2013:
                metadata['actressactive_when'] = 'modern'
            else:
                metadata['actressactive_when'] = 'current'
        
        if re.search(r'\bMadeupname\b', filename):
            metadata['madeupname'] = True
        
        return metadata

    def extract_file_metadata_rating(self, video_path):
        """Extract rating from video file metadata using ffprobe"""
        try:
            probe_cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_format', video_path
            ]
            
            result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                
                if 'format' in data and 'tags' in data['format']:
                    tags = data['format']['tags']
                    
                    # Check various possible rating tag names
                    rating_keys = ['rating', 'RATING', 'Rating', 'rate', 'RATE']
                    
                    for key in rating_keys:
                        if key in tags:
                            rating_value = tags[key]
                            
                            # Try to parse numeric rating (0-5 or 0-100)
                            try:
                                rating_num = float(rating_value)
                                
                                # If 0-5 scale
                                if 0 <= rating_num <= 5:
                                    return self.map_numeric_rating_to_stars(rating_num)
                                # If 0-100 scale (percentage)
                                elif 0 <= rating_num <= 100:
                                    normalized = rating_num / 20  # Convert to 0-5
                                    return self.map_numeric_rating_to_stars(normalized)
                            except (ValueError, TypeError):
                                pass
            
            return None, None
            
        except Exception as e:
            return None, None
    
    def map_numeric_rating_to_stars(self, rating_num):
        """Map numeric rating (0-5) to text rating and stars"""
        if rating_num >= 4.5:
            return 'OMG', '⭐⭐⭐⭐⭐'
        elif rating_num >= 3.5:
            return 'WOW', '⭐⭐⭐⭐'
        elif rating_num >= 2.5:
            return 'hmmm', '⭐⭐⭐'
        elif rating_num >= 1.5:
            return 'hmm', '⭐⭐'
        elif rating_num >= 0.5:
            return 'meh', '⭐'
        else:
            return None, None
    
    def extract_video_metadata(self, video_path):
        """Extract technical video metadata using ffprobe"""
        try:
            probe_cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_format', '-show_streams', video_path
            ]
            
            result = subprocess.run(probe_cmd, capture_output=True, text=True, encoding='utf-8', timeout=15)
            
            if result.returncode != 0:
                return {}
            
            data = json.loads(result.stdout)
            metadata = {}
            
            # Get format information
            if 'format' in data:
                format_info = data['format']
                
                # Container format
                if 'format_name' in format_info:
                    formats = format_info['format_name'].split(',')
                    metadata['format'] = formats[0].upper()
                
                # File size in MB (numeric)
                if 'size' in format_info:
                    size_bytes = int(format_info['size'])
                    metadata['file_size_mb'] = round(size_bytes / (1024 * 1024), 2)
                
                # Duration (convert to readable format)
                if 'duration' in format_info:
                    duration_sec = float(format_info['duration'])
                    metadata['duration'] = self.format_duration(duration_sec)
                
                # Bitrate
                if 'bit_rate' in format_info:
                    bitrate = int(format_info['bit_rate'])
                    metadata['bitrate'] = bitrate // 1000
                
                # Comments / Description from tags
                if 'tags' in format_info and isinstance(format_info['tags'], dict):
                    tags = format_info['tags']
                    comment_keys = ['comment', 'COMMENT', 'description', 'Description', 'DESCRIPTION', 'synopsis', 'Synopsis', 'summary', 'Summary', 'SUMMARY', 'notes', 'Notes', 'NOTES']
                    comments = []
                    for key in comment_keys:
                        if key in tags and tags[key]:
                            clean_comment = tags[key].strip()
                            if clean_comment:
                                comments.append(clean_comment)
                    if comments:
                        # Deduplicate while preserving order
                        seen = set()
                        ordered = []
                        for c in comments:
                            if c not in seen:
                                seen.add(c)
                                ordered.append(c)
                        metadata['file_comments'] = ' | '.join(ordered)
            
            # Get video and audio stream information
            if 'streams' in data:
                for stream in data['streams']:
                    codec_type = stream.get('codec_type', '')
                    
                    # Video stream
                    if codec_type == 'video' and 'video_codec' not in metadata:
                        # Video codec
                        if 'codec_name' in stream:
                            codec = stream['codec_name'].upper()
                            # Friendly names
                            codec_map = {
                                'H264': 'H.264', 'H265': 'H.265', 'HEVC': 'H.265',
                                'VP8': 'VP8', 'VP9': 'VP9', 'AV1': 'AV1',
                                'MPEG4': 'MPEG-4', 'XVID': 'XviD', 'DIVX': 'DivX'
                            }
                            metadata['video_codec'] = codec_map.get(codec, codec)
                        
                        # Resolution (width/height numeric)
                        if 'width' in stream and 'height' in stream:
                            metadata['width'] = int(stream['width'])
                            metadata['height'] = int(stream['height'])
                        
                        # Aspect ratio (decimal)
                        if 'display_aspect_ratio' in stream:
                            dar = str(stream['display_aspect_ratio'])
                            m = re.match(r'^(\d+)[:](\d+)$', dar)
                            if m:
                                w = int(m.group(1)); h = int(m.group(2))
                                if h:
                                    metadata['aspect_ratio'] = round(w / h, 2)
                        elif 'width' in stream and 'height' in stream:
                            w = int(stream.get('width', 0)); h = int(stream.get('height', 0))
                            if h:
                                metadata['aspect_ratio'] = round(w / h, 2)
                        
                        # Frame rate
                        if 'r_frame_rate' in stream:
                            fps_str = stream['r_frame_rate']
                            if '/' in fps_str:
                                num, den = map(int, fps_str.split('/'))
                                if den != 0:
                                    fps = round(num / den, 2)
                                    metadata['frame_rate'] = f"{fps} fps"
                    
                    # Audio stream
                    elif codec_type == 'audio' and 'audio_codec' not in metadata:
                        if 'codec_name' in stream:
                            codec = stream['codec_name'].upper()
                            # Friendly names
                            codec_map = {
                                'AAC': 'AAC', 'MP3': 'MP3', 'OPUS': 'Opus',
                                'VORBIS': 'Vorbis', 'AC3': 'AC-3', 'EAC3': 'E-AC-3',
                                'DTS': 'DTS', 'FLAC': 'FLAC', 'PCM_S16LE': 'PCM'
                            }
                            metadata['audio_codec'] = codec_map.get(codec, codec)
            
            return metadata
            
        except Exception as e:
            self.log(f"  ⚠️ Failed to extract video metadata: {str(e)}", "warning")
            return {}
    
    def format_duration(self, seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}{minutes:02d}{secs:02d}"
    
    def extract_tags(self, text, actress_name=''):
        tags = set()
        
        # Create a set of words from actress name to exclude
        actress_words = set()
        if actress_name:
            actress_words = {word.lower() for word in actress_name.split()}
        
        # IMPORTANT: Only extract tags from actress description, not partner/scene descriptions
        # Stop at action verbs to avoid tagging partner descriptors
        tag_search_text = text
        action_patterns = [
            r'\s+(?:gets?|sucks?|sucks? off|fucks?|fucked|does|is|shows?|performs?|had to try|plays?|dances?|interviews?|strips?|takes?|gives?)\s+',
            r'\s+(?:by|from)\s+(?!her|his|their|the|a|an)\s*',
        ]
        
        for pattern in action_patterns:
            match = re.search(pattern, tag_search_text, re.IGNORECASE)
            if match:
                tag_search_text = tag_search_text[:match.start()]
                break
        
        # Legacy tag_hierarchies removed: consolidation handles generalization
        
        
        tag_consolidation = {
            # Pigtails variations
            'pigtails': 'pigtailed',
            'pig-tailed': 'pigtailed',
            
            # Tattoo variations
            'tattoo': 'tattooed',
            'tattooed': 'tattooed',
            'tattoos': 'tattooed',
            'with-tattoos': 'tattooed',
            'dragon-tattoo': 'tattooed',
            'scorpion-tattoo': 'tattooed',
            'tiger-tattoo': 'tattooed',
            'rabbit-tattoo': 'tattooed',
            'rose-tattoo': 'tattooed',
            'heart-tattoo': 'tattooed',
            'heart-and-arrow-tattoo': 'tattooed',
            'cupid-tattoo': 'tattooed',
            'butterfly-tattoo': 'tattooed',
            'tribal-tattoo': 'tattooed',
            'snake-tattoo': 'tattooed',
            'devil-tattoo': 'tattooed',
            'angel-tattoo': 'tattooed',
            'cross-tattoo': 'tattooed',
            'playboy-tattoo': 'tattooed',
            'clown-tattoo': 'tattooed',
            'cherub-tattoo': 'tattooed',
            'dolphin-tattoo': 'tattooed',
            'small-tattoos': 'tattooed',
            'small-heart-tattoo': 'tattooed',

            # Piercing variations
            'pierced': 'pierced',
            'nipple-pierced': 'pierced',
            'navel-pierced': 'pierced',
            'clit-pierced': 'pierced',
            
            # Hair variations
            'dyed-haired': 'dyed-hair',
            'dyed-blond': 'blonde',
            'dyed-blonde': 'blonde',
            'frizzy-haired': 'frizzy-hair',
            'short-ish-haired': 'short-haired',
            
            # Glasses variations
            'glasses-wearing': 'glasses',
            'with-glasses': 'glasses',
            'without-glasses': 'no-glasses',  # Keep as separate tag
            
            # Hairy variations (specific -> general)
            'hairy-cunny': 'hairy',
            'hairy-pussy': 'hairy',
            'hairy-assed': 'hairy',
            'hairy-armpits': 'hairy',
            'hairy-legs': 'hairy',
            'hairy-legged': 'hairy',
            'full-bush': 'hairy',
            'slightly-hairy': 'hairy',
            'unshaved': 'hairy',
            'hirsute': 'hairy',

            # Hairy ass needs to be separate (needs extra emphasis)
            'hairy-assed': 'hairy-ass',
                        
            # Body type consolidation
            'plumper': 'chubby',
            'plump': 'chubby',
            'chunky': 'chubby',
            'fattier': 'fat',
            'chubbier': 'chubby',
            'BBW': 'BBW',  # Keep separate (more extreme)
            'skinny': 'thin',
            'slender': 'slim',
            'child-bodied': 'small-bodied',
            'small-body': 'small-bodied',
            'small-bodied': 'small-bodied',
            'small-framed': 'small-bodied',
            'tiny': 'small-bodied',
            'petite': 'small-bodied',

            # Breast size consolidation
            'huge-titted': 'big-titted',
            'nice-titted': 'big-titted',
            'huge-saggy': 'saggy-titted',
            'great-saggy-tits': 'saggy-titted',
            'huge-boobs': 'big-titted',
            'huge-breasted': 'big-titted',
            'large-breasted': 'big-titted',
            'nice-tits': 'big-titted',
            'great-tits': 'big-titted',
            'fantastic-tits': 'big-titted',
            'big-tits': 'big-titted',
            'busty': 'big-titted',
            'fabulous-big-tits': 'big-titted',
            'torpedo-tits': 'big-titted',
            'soft-titted': 'big-titted',
            'flat-chested': 'small-titted',
            'flat-chest': 'small-titted',
            'fake-titted': 'fake-titted',  # Keep separate
            'fake-tits': 'fake-titted',
            'pumped-up-tits': 'fake-titted',
            'pumped-tits': 'fake-titted',
            
            # Skin variations
            'creamy-white-skinned': 'pale-skinned',
            'white-skinned': 'pale-skinned',
            'nice-white-skin': 'pale-skinned',
            'clear-skin': 'nice-skin',
            'bad-facial-skin': 'acned',
            'pimpled-ass': 'pimpled',

            # Moles and marks
            'mole-on-face': 'mole',
            'mole': 'mole',
            'moles-on-chest': 'mole',
            'moles-on-body': 'mole',
            'birthmark': 'birthmark',  # Keep separate (different from mole)
            'beauty-mark': 'mole',
            'beauty-marks': 'mole',
            
            # Attractiveness consolidation
            'sensational': 'gorgeous',
            'amazing': 'gorgeous',
            'fantastic': 'gorgeous',
            'stunning': 'gorgeous',
            'beautiful': 'beautiful',
            'attractive': 'attractive',
            'foxy': 'pretty',
            'sweet': 'pretty',
            'elegant': 'pretty',
            'innocent': 'pretty',
            'average-pretty': 'average',
            'ugly-pretty': 'average',
            
            # Personality/style
            'next-door': 'average',
            'plain-looking': 'plain',
            'whorish': 'skanky',
            
            # Ethnic/appearance
            'roma': 'roma',
            'half-roma': 'half-roma',
            'mestiza': 'mixed-race',
            'mixed': 'mixed-race',
            'latina': 'latina',
            'native': 'native',
            'ebony': 'ebony',
            
            # Age-related
            'young-looking': 'young',
            'younger': 'young',
            'teenie': 'teen',
            'daughter': 'teen',
            'older': 'mature',
            'mom': 'mature',
            'mother': 'mature',
            'wife': 'mature',
            'housewife': 'mature',
            'preggo': 'pregnant',
            'post-natal': 'post-natal',
            
            # Nose features (consolidate similar)
            'hook-nose': 'big-nose',
            'crooked-nose': 'big-nose',
            'ugly-nose': 'big-nose',
            'pointed-nose': 'pointed-nose',  # Keep specific
            'button-nose': 'button-nose',  # Keep specific
            'curved-nose': 'big-nose',
            
            # Teeth features
            'strange-teeth': 'uneven-teeth',
            'uneven-front-teeth': 'uneven-teeth',
            'large-teeth': 'uneven-teeth',
            'gaps-between-teeth': 'gaps-between-teeth',
            'toothless': 'toothless',
            
            # Face features
            'duck-faced': 'duck-faced',  # Keep as-is
            'pixie-face': 'pixie-face',  # Keep as-is
            'wide-face': 'wide-face',  # Keep as-is
            'flat-faced': 'flat-faced',  # Keep as-is
            
            # Eyes
            'gorgeous-eyes': 'beautiful-eyes',
            'large-eyed': 'beautiful-eyes',
            
            # Chin
            'large-chin': 'large-chin',  # Keep specific
            'weak-chin': 'weak-chin',  # Keep specific
            
            # Ass quality
            'good-assed': 'nice-ass',
            'great-ass': 'nice-ass',
            'fantastic-ass': 'nice-ass',
            'gorgeous-ass': 'nice-ass',
            'big-assed': 'big-assed',  # Keep separate
            'huge-ass': 'big-assed',
            'bubble-butt-ass': 'big-assed',
            
            # Body quality
            'nice-body': 'good-body',
            'great-body': 'good-body',
            'gorgeous-body': 'good-body',
            'perfect-body': 'good-body',
            
            # Pussy descriptors
            'meaty-pussy-lips': 'big-pussy-lips',
            'huge-pussy-lips': 'big-pussy-lips',
            
            # Era
            'vintage': 'retro',
            'classic': 'retro',
            
            # Quality terms
            'babe': 'pretty',
            'beauty': 'pretty',
            'hottie': 'pretty',
            'sweetie': 'pretty',
            'cutie': 'pretty',
            'chick': 'average',
            
            # Production
            'home-made': 'homemade',
            'pro-am': 'semi-pro',
            
            # Special
            'all-time-fave': 'fave',
            'negro-lover': 'BBC-lover',
            
            # Misc
            'nerdy': 'nerdy',  # Keep as-is
            'emo': 'emo',  # Keep as-is
            'rare': 'rare',  # Keep as-is
            'Goddess': 'gorgeous',
            'Neanderthal': 'ugly',
        }
        
        potential_tags = set()
        for keyword in TAG_KEYWORDS:
            # Skip if keyword is part of actress name
            if keyword.lower() in actress_words:
                continue
                
            search_pattern = keyword.replace('-', r'[\s-]')
            if re.search(fr'\b{search_pattern}\b', text, re.IGNORECASE):
                canonical_tag = tag_consolidation.get(keyword, keyword)
                potential_tags.add(canonical_tag)
        
        # Hierarchy cleanup removed; consolidation ensures canonical forms
        
        
        if ('tattoo' in potential_tags) or any(t.endswith('-tattoo') for t in potential_tags) or ('tattoos' in potential_tags) or ('tattooed' in potential_tags):
            potential_tags = {t for t in potential_tags if not (t == 'tattoo' or t == 'tattoos' or t.endswith('-tattoo'))}
            potential_tags.add('tattooed')
        
        if re.search(r'\bKURIÓZUM\b', text, re.IGNORECASE):
            potential_tags.add('kuriózum')
        
        return sorted(list(potential_tags))
    
    def extract_partners(self, text):
        # Remove everything from last "in" onwards (likely movie title)
        last_in_pos = text.rfind(' in ')
        if last_in_pos > 0:
            search_text = text[:last_in_pos]
        else:
            search_text = text
        
        partners = []
        sorted_partners = sorted(self.known_partners, key=len, reverse=True)
        matched_positions = set()
        
        for partner in sorted_partners:
            pattern = fr'(?<!AKA\s)\b{re.escape(partner)}\b(?!\s+AKA)'
            for match in re.finditer(pattern, search_text):
                start, end = match.span()
                if not any(p[0] <= start < p[1] or p[0] < end <= p[1] for p in matched_positions):
                    partners.append(partner)
                    matched_positions.add((start, end))
        
        return sorted(partners)
    
    def generate_short_filename(self, metadata, sequence_num):
        actress_uuid = metadata['actress_uuid']
        
        parts = [actress_uuid, f"{sequence_num:03d}"]
        
        if metadata['movie']:
            movie_code = self.generate_movie_code(metadata['movie'])
            if movie_code:
                parts.append(movie_code)
        
        if metadata['rating']:
            parts.append(metadata['rating'])
        
        return '_'.join(parts)

    def create_actress_folder_and_move(self, video_meta, actress_alias):
        """Create actress folder based on alias and move video file there"""
        try:
            current_path = video_meta['current_path']
            parent_dir = os.path.dirname(current_path)
            
            # Check if _pic_sets or _pic_set folder exists in current directory
            pic_sets_exists = os.path.exists(os.path.join(parent_dir, '_pic_sets')) or \
                             os.path.exists(os.path.join(parent_dir, '_pic_set'))
            
            if pic_sets_exists:
                self.log(f"  ✓ _pic_sets folder detected - keeping files in current location")
                return current_path, parent_dir
            
            # Check if already in a matching folder
            current_folder_name = os.path.basename(parent_dir)
            
            # Simple fuzzy match: check if folder name starts with actress name
            actress_first_name = video_meta['actress_name'].split()[0]
            if current_folder_name.startswith(actress_first_name):
                self.log(f"  ✓ Already in actress folder: {current_folder_name}")
                return current_path, parent_dir
            
            # Create new folder with full alias
            actress_folder = os.path.join(parent_dir, actress_alias)
            os.makedirs(actress_folder, exist_ok=True)
            
            # Move file to new folder
            new_path = os.path.join(actress_folder, os.path.basename(current_path))
            shutil.move(current_path, new_path)
            
            self.log(f"  📁 Created/moved to: {actress_alias}/", "success")
            return new_path, actress_folder
            
        except Exception as e:
            self.log(f"  ⚠️ Folder move failed: {str(e)}", "warning")
            return video_meta['current_path'], os.path.dirname(video_meta['current_path'])

    def move_subtitle_files(self, old_video_path, new_video_path, original_base=None):
        """Find and move subtitle files with the same base name as the video"""
        subtitle_extensions = {'.srt', '.ass', '.ssa', '.vtt', '.sub', '.idx', '.smi', '.sami'}
        
        old_dir = os.path.dirname(old_video_path)
        new_dir = os.path.dirname(new_video_path)
        
        old_basename = os.path.splitext(os.path.basename(old_video_path))[0]
        new_basename = os.path.splitext(os.path.basename(new_video_path))[0]
        original_basename = original_base if original_base else old_basename
        
        moved_subtitles = []
        
        # Look for subtitle files with the same base name
        for file in os.listdir(old_dir):
            file_path = os.path.join(old_dir, file)
            
            # Skip if not a file
            if not os.path.isfile(file_path):
                continue
            
            file_base, file_ext = os.path.splitext(file)
            
            # Check if it's a subtitle file with matching base name
            if file_ext.lower() in subtitle_extensions and (file_base == old_basename or file_base == original_basename):
                # Create new subtitle path
                new_subtitle_name = new_basename + file_ext
                new_subtitle_path = os.path.join(new_dir, new_subtitle_name)
                
                try:
                    shutil.move(file_path, new_subtitle_path)
                    moved_subtitles.append((file, new_subtitle_name))
                    self.log(f"  📄 Moved subtitle: {file} → {new_subtitle_name}", "success")
                except Exception as e:
                    self.log(f"  ⚠️ Failed to move subtitle {file}: {str(e)}", "warning")
        
        return len(moved_subtitles) > 0  # Return True if any subtitles were moved

    def create_reference_txt_file(self, video_meta, actress_folder):
        """Create empty txt file with original filename for search purposes"""
        try:
            original_filename = os.path.splitext(video_meta['original_filename'])[0]
            txt_filename = f"{original_filename}.txt"
            txt_path = os.path.join(actress_folder, txt_filename)
            
            # Create empty file
            with open(txt_path, 'w', encoding='utf-8') as f:
                pass  # Empty file
            
            self.log(f"  📄 Created reference: {txt_filename}", "success")
            return True
            
        except Exception as e:
            self.log(f"  ⚠️ Reference file creation failed: {str(e)}", "warning")
            return False
        
    def encode_file_uri(self, path):
        path = os.path.abspath(path).replace('\\', '/')
        if IS_WINDOWS and len(path) > 1 and path[1] == ':':
            drive = path[0:2]
            rest = path[2:]
            encoded_rest = urllib.parse.quote(rest)
            return f"file:///{drive}{encoded_rest}"
        encoded_path = urllib.parse.quote(path)
        return f"file:///{encoded_path}"
    
    def encode_network_url(self, path, host='192.168.0.101', port='8080'):
        abs_path = os.path.abspath(path)
        rel = re.sub(r'^[A-Za-z]:\\', '', abs_path)
        rel = rel.replace('\\', '/')
        parts = [urllib.parse.quote(p) for p in rel.split('/') if p]
        return f"vlc://http://{host}:{port}/" + '/'.join(parts)
    
    def generate_movie_code(self, movie_title):
        if not movie_title:
            return ""
        
        clean_title = re.sub(r'\([^)]*\)', '', movie_title)
        words = [w for w in clean_title.split() if re.match(r'[A-Za-z]', w) and len(re.sub(r'[^A-Za-z]', '', w)) > 2]
        code = ''.join(re.sub(r'[^A-Za-z]', '', w)[0].upper() for w in words)
        numbers = ''.join(c for c in movie_title if c.isdigit())
        
        return (code + numbers)[:10]
    
    def generate_contact_sheet(self, video_path, output_path, width=260, cols=4, rows=2):
        try:
            duration_cmd = [
                'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1', video_path
            ]
            result = subprocess.run(duration_cmd, capture_output=True, text=True, timeout=30)
            duration = float(result.stdout.strip())
            
            num_thumbs = cols * rows
            thumb_height = int(width * 9 / 16)
            
            fps_rate = (num_thumbs + 1) / max(duration, 1.0)
            filter_complex = (
                f"fps={fps_rate},"
                f"scale={width}:{thumb_height}:force_original_aspect_ratio=decrease,"
                f"pad={width}:{thumb_height}:(ow-iw)/2:(oh-ih)/2:black,"
                f"tile={cols}x{rows}"
            )
            
            thumbnail_cmd = [
                'ffmpeg', '-i', video_path,
                '-filter_complex', filter_complex,
                '-frames:v', '1',
                '-y', output_path
            ]
            
            subprocess.run(thumbnail_cmd, capture_output=True, timeout=90, check=True)
            
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                return True
            
            return False
            
        except Exception as e:
            self.log(f"  ⚠️ Contact sheet generation failed: {str(e)}", "warning")
            return False
    
    def extract_poster_image(self, video_path, output_path):
        try:
            probe_cmd = [
                'ffprobe', '-v', 'quiet', '-select_streams', 'v',
                '-show_entries', 'stream=index:stream=codec_name:stream_disposition=attached_pic',
                '-of', 'json', video_path
            ]
            
            result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                
                if 'streams' in data:
                    for stream in data['streams']:
                        if 'disposition' in stream and stream['disposition'].get('attached_pic') == 1:
                            stream_index = stream['index']
                            
                            extract_cmd = [
                                'ffmpeg', '-i', video_path, '-map', f'0:{stream_index}',
                                '-c', 'copy', '-y', output_path
                            ]
                            
                            subprocess.run(extract_cmd, capture_output=True, timeout=60, check=True)
                            
                            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                                return True
            
            return False
            
        except Exception as e:
            return False
    
    def generate_card_thumbnail(self, video_path, output_path, width=256):
        try:
            if self.extract_poster_image(video_path, output_path):
                return True
            
            duration_cmd = [
                'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1', video_path
            ]
            result = subprocess.run(duration_cmd, capture_output=True, text=True, timeout=30)
            duration = float(result.stdout.strip())
            timestamp = duration * 0.1
            
            thumbnail_cmd = [
                'ffmpeg', '-ss', str(timestamp), '-i', video_path,
                '-vframes', '1', '-vf', f'scale={width}:-1',
                '-y', output_path
            ]
            
            subprocess.run(thumbnail_cmd, capture_output=True, timeout=60, check=True)
            
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                return True
            
            return False
            
        except Exception as e:
            return False
    
    def read_existing_markdown(self, file_path):
        data = {}
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if content.startswith('---'):
                    _, frontmatter, _ = content.split('---', 2)
                    current_key = None
                    for line in frontmatter.strip().split('\n'):
                        # Check if this is a key:value line (not indented, has colon)
                        if ':' in line and not line.startswith('  '):
                            key, value = line.split(':', 1)
                            key = key.strip()
                            value = value.strip()
                            
                            # If value is empty, this is a list property
                            if value == '':
                                data[key] = []
                                current_key = key
                            else:
                                data[key] = value
                                current_key = None
                        # Check if this is a list item (starts with '  - ')
                        elif line.startswith('  - ') and current_key:
                            item = line[4:].strip()
                            data[current_key].append(item)
                    return data, content
        except FileNotFoundError:
            return None, None
        return None, None
    
    def should_update_file(self, old_data, new_data):
        if not old_data:
            return True
        
        fields_to_compare = [
            'title', 'tags', 'madeupname', 'usualhaircolor', 'bodytype',
            'done_bbc', 'done_anim', 'nationality', 'actressactive_when',
            'partners_with'
        ]
        
        for field in fields_to_compare:
            old_value = old_data.get(field)
            new_value = new_data.get(field)
            if old_value != new_value:
                return True
        
        return False
    
    def process_videos(self):
        video_folder = self.video_folder.get()
        single_file = self.single_video_file.get()
        stash_base = self.stash_base.get()
        
        if not stash_base:
            messagebox.showerror("Error", "Please specify STASH base folder")
            return
        
        self.actress_database = {}
        self.video_metadata = []
        self.rename_log = []
        self.used_uuids = set()
        
        self.log("=" * 100)
        self.log("🚀 UNIFIED VIDEO PROCESSOR V2.5.2", "info")
        self.log(f"Mode: {'DRY RUN (Preview Only)' if self.dry_run.get() else 'LIVE (Will Rename Files)'}", 
                "warning" if not self.dry_run.get() else "info")
        self.log("=" * 100)
        
        self.load_partners()
        
        self.actress_index = {}
        self.video_index = {}
        self.load_actress_index(stash_base)
        self.load_video_index(stash_base)
        self.log(f"  📚 Loaded actress index with {len(self.actress_index.get('aliases', []))} aliases")
        self.log(f"  📹 Loaded video index with {len(self.video_index.get('aliases', []))} aliases")
        
        self.progress.start()
        self.status_var.set("🔍 Scanning videos...")
        
        video_extensions = {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.mpg', '.mpeg', '.m4v', '.webm', '.3gp', '.ogv', '.asf', '.asx', '.vob', '.divx', '.f4v', '.rm', '.rmvb'}
        video_files = []
        
        if self.selected_video_files:
            video_files = [f for f in self.selected_video_files if os.path.exists(f) and Path(f).suffix.lower() in video_extensions]
            if not video_files:
                messagebox.showerror("Error", "Please select valid video files")
                self.progress.stop()
                return
        elif single_file:
            if os.path.exists(single_file) and Path(single_file).suffix.lower() in video_extensions:
                video_files = [single_file]
            else:
                messagebox.showerror("Error", "Please select a valid video file")
                self.progress.stop()
                return
        else:
            if not video_folder:
                messagebox.showerror("Error", "Please select a valid video folder or pick file(s)")
                self.progress.stop()
                return
            
            # Handle multiple folders (separated by '; ')
            folders = [f.strip() for f in video_folder.split(';')]
            
            for folder in folders:
                if not os.path.exists(folder):
                    self.log(f"⚠️ Folder not found: {folder}", "warning")
                    continue
                
                self.log(f"📁 Scanning folder: {folder}", "info")
                for root_dir, dirs, files in os.walk(folder):
                    for file in files:
                        if Path(file).suffix.lower() in video_extensions:
                            video_files.append(os.path.join(root_dir, file))
        
        self.log(f"📁 Found {len(video_files)} video files", "success")
        
        self.log("\n" + "=" * 100)
        self.log("PHASE 1: Extracting Metadata", "info")
        self.log("=" * 100)
        self.status_var.set("📋 Phase 1: Extracting metadata...")
        
        actress_videos = defaultdict(list)
        skipped_files = 0
        
        for idx, video_path in enumerate(video_files, 1):
            if idx % 10 == 0:
                self.status_var.set(f"📋 Phase 1: Processing {idx}/{len(video_files)}...")
                self.root.update()
            
            filename = os.path.basename(video_path)
            
            if filename.startswith('STA_'):
                self.log(f"⚠️ Skipping (STA_ prefix): {filename}", "warning")
                skipped_files += 1
                continue
            
            # Skip files that already have UUID pattern (already processed)
            if re.match(r'^[A-Z][a-z]{2}[A-Z0-9]{5}_\d{3}_', filename):
                self.log(f"⚠️ Skipping (already processed): {filename}", "warning")
                skipped_files += 1
                continue
            
            # Quick check: skip files without nationality
            if not re.search(fr'\b({NATIONALITY_PATTERN})\b', filename):
                self.log(f"⚠️ Skipping (no nationality found): {filename}", "warning")
                skipped_files += 1
                continue
            
            metadata = self.parse_filename_metadata(filename)
            metadata['original_path'] = video_path
            metadata['extension'] = Path(filename).suffix
            
            # If no rating found in filename, try extracting from file metadata
            if not metadata['rating']:
                file_rating, file_stars = self.extract_file_metadata_rating(video_path)
                if file_rating:
                    metadata['rating'] = file_rating
                    metadata['rating_stars'] = file_stars
                    self.log(f"  📊 Extracted rating from metadata: {file_rating} {file_stars}")
            
            # Extract video technical metadata
            video_tech_metadata = self.extract_video_metadata(video_path)
            metadata.update(video_tech_metadata)
            
            if metadata['actress_name']:
                actress_videos[metadata['actress_name']].append(metadata)
                self.video_metadata.append(metadata)
            else:
                self.log(f"⚠️ Skipping (no actress name): {filename}", "warning")
                skipped_files += 1
        
        self.log(f"✅ Extracted metadata from {len(self.video_metadata)} videos", "success")
        self.log(f"✅ Found {len(actress_videos)} unique actresses", "success")
        if skipped_files > 0:
            self.log(f"⚠️ Skipped {skipped_files} files", "warning")
        
        self.log("\n" + "=" * 100)
        self.log("PHASE 2: Generating Smart UUIDs & Short Filenames", "info")
        self.log("=" * 100)
        self.status_var.set("🔤 Phase 2: Generating UUIDs and filenames...")
        
        for actress_name, videos in actress_videos.items():
            existing_uuid = self.find_existing_actress_uuid(videos[0]['original_filename'], stash_base)
            actress_uuid = existing_uuid if existing_uuid else self.generate_actress_uuid(actress_name)
            videos.sort(key=lambda x: x['original_filename'])
            
            # Build title_with_descriptors for display in dry run
            first_video = videos[0]
            
            # DEBUG: Log what descriptors we have
            self.log(f"  DEBUG: First video descriptors: {first_video.get('descriptors', [])}")
            for i, v in enumerate(videos):
                self.log(f"  DEBUG: Video {i+1} descriptors: {v.get('descriptors', [])}")
            
            title_with_descriptors = actress_name
            noun_desc = self.get_title_noun_descriptor(videos)
            if first_video['nationality'] and (noun_desc or first_video.get('descriptors')):
                title_with_descriptors += f" {first_video['nationality']}"
            if noun_desc:
                title_with_descriptors += f" {noun_desc}"
            
            # Log actress info with full title
            self.log(f"\n{'='*80}")
            self.log(f"  Actress: {actress_name}")
            self.log(f"  UUID: {actress_uuid}")
            self.log(f"  Full Title: {title_with_descriptors}", "success")
            self.log(f"  Videos: {len(videos)}")
            self.log(f"{'='*80}")
            
            for seq_num, video_meta in enumerate(videos, 1):
                video_meta['actress_uuid'] = actress_uuid
                short_name = self.generate_short_filename(video_meta, seq_num)
                video_meta['short_filename'] = short_name + video_meta['extension']
                video_meta['sequence_number'] = seq_num
                
                self.log(f"  [{seq_num:02d}] {video_meta['original_filename'][:60]}...")
                self.log(f"      → {video_meta['short_filename']}", "success")
                
                # Show video frontmatter preview in dry run
                if self.dry_run.get():
                    self.log(f"      Movie: {video_meta['movie']}")
                    if video_meta['movie_alt']:
                        self.log(f"      Movie Alt: {', '.join(video_meta['movie_alt'])}")
                    if video_meta['version']:
                        self.log(f"      Version: {video_meta['version']}")
                    if video_meta.get('aie'):
                        self.log(f"      AIE: True")
                    if video_meta.get('bracket_info'):
                        self.log(f"      Notes: {'; '.join(video_meta['bracket_info'][:2])}{'...' if len(video_meta['bracket_info']) > 2 else ''}")
                    if video_meta['rating']:
                        self.log(f"      Rating: {video_meta['rating']} {video_meta['rating_stars']}")
                    if video_meta['partners']:
                        self.log(f"      Partners: {', '.join(video_meta['partners'][:3])}{'...' if len(video_meta['partners']) > 3 else ''}")
                    self.log("")  # Blank line for readability
        
        if not self.dry_run.get():
            self.log("\n" + "=" * 100)
            self.log("PHASE 3: Renaming Files", "warning")
            self.log("=" * 100)
            self.status_var.set("📝 Phase 3: Renaming files...")
            
            # Group videos by actress for folder organization
            actress_video_groups = defaultdict(list)
            for video_meta in self.video_metadata:
                actress_video_groups[video_meta['actress_uuid']].append(video_meta)
            
            for idx, video_meta in enumerate(self.video_metadata, 1):
                if idx % 5 == 0:
                    self.status_var.set(f"📝 Phase 3: Renaming {idx}/{len(self.video_metadata)}...")
                    self.root.update()
                
                old_path = video_meta['original_path']
                new_filename = video_meta['short_filename']
                new_path = os.path.join(os.path.dirname(old_path), new_filename)
                
                try:
                    # Step 1: Rename file
                    os.rename(old_path, new_path)
                    video_meta['current_path'] = new_path
                    self.log(f"✅ Renamed: {os.path.basename(old_path)} → {new_filename}", "success")
                    
                    # Step 2: Build actress alias
                    actress_uuid = video_meta['actress_uuid']
                    actress_name = video_meta['actress_name']
                    videos_for_actress = actress_video_groups[actress_uuid]
                    
                    # Build full alias like in create_actress_note
                    full_alias = actress_name
                    if video_meta['nationality']:
                        full_alias += f" {video_meta['nationality']}"
                    
                    # Get common descriptors for this actress
                    common_descriptors = self.get_common_descriptors(videos_for_actress)
                    if common_descriptors:
                        full_alias += " " + " ".join(common_descriptors[:5])
                    
                    # Step 3: Move to actress folder (store old path before move)
                    old_video_location = new_path
                    final_path, actress_folder = self.create_actress_folder_and_move(video_meta, full_alias)
                    video_meta['current_path'] = final_path
                    
                    # Step 4: Move subtitle files from old location to actress folder
                    has_captions = self.move_subtitle_files(old_video_location, final_path, os.path.splitext(video_meta['original_filename'])[0])
                    video_meta['has_captions'] = has_captions
                    
                    # Step 5: Create reference txt file with original filename
                    self.create_reference_txt_file(video_meta, actress_folder)
                    
                    self.rename_log.append({'old': old_path, 'new': final_path})
                    
                except Exception as e:
                    self.log(f"❌ Rename failed: {str(e)}", "error")
                    video_meta['current_path'] = old_path
        else:
            self.log("\n⚠️ SKIPPING Phase 3: Dry Run mode - no files renamed", "warning")
            for video_meta in self.video_metadata:
                video_meta['current_path'] = video_meta['original_path']
        
        self.log("\n" + "=" * 100)
        self.log("PHASE 4: Generating Thumbnails", "info")
        self.log("=" * 100)
        
        if not self.dry_run.get():
            self.status_var.set("🎬 Phase 4: Generating thumbnails...")
            
            total_videos = len(self.video_metadata)
            processed = 0
            
            for actress_name, videos in actress_videos.items():
                actress_uuid = videos[0]['actress_uuid']
                first_char = actress_uuid[0].upper()
                
                resources_dir = os.path.join(stash_base, first_char, '_resources')
                os.makedirs(resources_dir, exist_ok=True)
                
                videos_resources_dir = os.path.join(stash_base, 'Videos', first_char, '_resources')
                os.makedirs(videos_resources_dir, exist_ok=True)
                
                self.log(f"\n📸 Generating thumbnails for {actress_name} ({actress_uuid})...")
                
                for video_meta in videos:
                    processed += 1
                    self.status_var.set(f"🎬 Phase 4: Thumbnails {processed}/{total_videos}...")
                    self.root.update()
                    
                    video_name = os.path.splitext(video_meta['short_filename'])[0]
                    
                    thumb_path = os.path.join(resources_dir, f"{video_name}_thumb.jpg")
                    
                    if os.path.exists(thumb_path):
                        self.log(f"  ✓ Contact sheet exists: {video_name}_thumb.jpg")
                        video_meta['thumbnail_path'] = thumb_path
                    else:
                        self.log(f"  🔄 Generating contact sheet: {video_name}_thumb.jpg")
                        current_path = video_meta['current_path']
                        if self.generate_contact_sheet(current_path, thumb_path):
                            self.log(f"  ✅ Generated: {video_name}_thumb.jpg", "success")
                            video_meta['thumbnail_path'] = thumb_path
                        else:
                            self.log(f"  ❌ Failed: {video_name}_thumb.jpg", "error")
                            video_meta['thumbnail_path'] = None
                    
                    card_path = os.path.join(videos_resources_dir, f"{video_name}_card.jpg")
                    
                    if os.path.exists(card_path):
                        self.log(f"  ✓ Card exists: {video_name}_card.jpg")
                        video_meta['card_path'] = card_path
                    else:
                        self.log(f"  🔄 Generating card: {video_name}_card.jpg")
                        current_path = video_meta['current_path']
                        if self.generate_card_thumbnail(current_path, card_path):
                            self.log(f"  ✅ Generated: {video_name}_card.jpg", "success")
                            video_meta['card_path'] = card_path
                        else:
                            self.log(f"  ❌ Failed: {video_name}_card.jpg", "error")
                            video_meta['card_path'] = None
        else:
            self.log("⚠️ SKIPPING Phase 4: Dry Run mode", "warning")
        
        self.log("\n" + "=" * 100)
        self.log("PHASE 5: Creating Obsidian Notes", "info")
        self.log("=" * 100)
        
        if not self.dry_run.get():
            self.status_var.set("📝 Phase 5: Creating notes...")
            
            for idx, (actress_name, videos) in enumerate(actress_videos.items(), 1):
                self.status_var.set(f"📝 Phase 5: Notes {idx}/{len(actress_videos)}...")
                self.root.update()
                
                self.create_actress_note(actress_name, videos, stash_base)
                
                for video_meta in videos:
                    self.create_video_note(video_meta, stash_base)
        else:
            self.log("⚠️ SKIPPING Phase 5: Dry Run mode", "warning")
        
        self.progress.stop()
        self.log("\n" + "=" * 100)
        self.log("✅ PROCESSING COMPLETE!", "success")
        self.log("=" * 100)
        
        if self.rename_log:
            self.log(f"\n📝 Rename log: {len(self.rename_log)} files renamed")
        
        self.status_var.set(f"✅ Complete: {len(self.video_metadata)} videos processed")
        
        messagebox.showinfo("Processing Complete", 
                          f"Processed {len(self.video_metadata)} videos\n"
                          f"Found {len(actress_videos)} actresses\n"
                          f"Mode: {'DRY RUN' if self.dry_run.get() else 'LIVE'}")
    
    def create_actress_note(self, actress_name, videos, stash_base):
        try:
            first_video = videos[0]
            actress_uuid = first_video['actress_uuid']
            
            all_tags = set()
            all_partners = set()
            done_bbc = False
            done_anim = False
            
            for video in videos:
                all_tags.update(video['tags'])
                all_partners.update(video['partners'])
                done_bbc |= video['has_bbc']
                done_anim |= video['has_anim']
            
            # Build title using common descriptors
            title_with_descriptors = actress_name
            noun_desc = self.get_title_noun_descriptor(videos)
            if first_video['nationality'] and (noun_desc or first_video.get('descriptors')):
                title_with_descriptors += f" {first_video['nationality']}"
            if noun_desc:
                title_with_descriptors += f" {noun_desc}"
            
            # Build full_alias for folder name (same as title)
            full_alias = title_with_descriptors
            
            first_char = actress_uuid[0].upper()
            dir_path = os.path.join(stash_base, first_char)
            os.makedirs(dir_path, exist_ok=True)
            
            file_path = os.path.join(dir_path, f"{actress_uuid}.md")
            
            old_data, old_content = self.read_existing_markdown(file_path)

            if old_data:
                existing_tags = set(old_data.get('tags', []))
                existing_partners = set(old_data.get('partners_with', []))
                all_tags = all_tags.union(existing_tags)
                all_partners = all_partners.union(existing_partners)
                done_bbc = done_bbc or (str(old_data.get('done_bbc', 'false')).lower() == 'true')
                done_anim = done_anim or (str(old_data.get('done_anim', 'false')).lower() == 'true')
                if not first_video["haircolor"]:
                    first_video["haircolor"] = old_data.get('usualhaircolor', '')
                if not first_video["bodytype"]:
                    first_video["bodytype"] = old_data.get('bodytype', '')
            
            new_data = {
                'title': actress_name,
                'tags': sorted(all_tags),
                'madeupname': first_video["madeupname"],
                'usualhaircolor': first_video["haircolor"],
                'bodytype': first_video["bodytype"],
                'done_bbc': done_bbc,
                'done_anim': done_anim,
                'nationality': first_video["nationality"],
                'actressactive_when': first_video["actressactive_when"],
                'partners_with': sorted(all_partners)
            }
            
            if not self.should_update_file(old_data, new_data):
                self.log(f"✓ No updates needed: {actress_uuid}.md", "info")
                return
            
            date_created = old_data.get('date_created') if old_data else datetime.now().strftime("%Y-%m-%dT%H:%M")
            date_modified = datetime.now().strftime("%Y-%m-%dT%H:%M")
            
            # Prepare updated frontmatter data
            updated_frontmatter = {}
            
            # Start with existing data to preserve custom properties
            if old_data:
                updated_frontmatter = old_data.copy()
            
            # Update only the properties we manage
            updated_frontmatter['title'] = title_with_descriptors
            
            # Build aliases list
            aliases = []
            aliases.append(actress_name)
            if first_video['actress_aliases']:
                aliases.extend(first_video['actress_aliases'])
                if self.include_title_in_aliases.get():
                    aliases.append(title_with_descriptors)
            else:
                if title_with_descriptors != actress_name:
                    aliases = [title_with_descriptors]
                else:
                    aliases = [actress_name]
            
            updated_frontmatter['aliases'] = aliases
            updated_frontmatter['tags'] = sorted(all_tags)
            updated_frontmatter['dg-publish'] = old_data.get('dg-publish', False) if old_data else False
            updated_frontmatter['madeupname'] = first_video["madeupname"]
            updated_frontmatter['usualhaircolor'] = first_video["haircolor"]
            updated_frontmatter['bodytype'] = first_video["bodytype"]
            updated_frontmatter['done_bbc'] = done_bbc
            updated_frontmatter['done_anim'] = done_anim
            updated_frontmatter['nationality'] = first_video["nationality"]
            updated_frontmatter['actressactive_when'] = first_video["actressactive_when"]
            updated_frontmatter['partners_with'] = sorted(all_partners)
            
            # Preserve existing external link values or set empty if new
            for field in ['iafd', 'egafd', 'bgafd', 'indexxx', 'vef']:
                if field not in updated_frontmatter:
                    updated_frontmatter[field] = old_data.get(field, '') if old_data else ''
            
            updated_frontmatter['obsidianUIMode'] = old_data.get('obsidianUIMode', 'preview') if old_data else 'preview'
            updated_frontmatter['date_created'] = date_created
            updated_frontmatter['date_modified'] = date_modified
            
            # Write frontmatter preserving all properties
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write('---\n')
                
                # Define property order matching Obsidian Linter configuration
                property_order = [
                    'title', 'aliases', 'tags', 'cssclasses', 'dg-publish', 'status',
                    'madeupname', 'usualhaircolor', 'bodytype', 'done_bbc', 'done_anim',
                    'nationality', 'actressactive_when', 'partners_with', 'rating',
                    'iafd', 'egafd', 'bgafd', 'indexxx', 'vef',
                    'head', 'attachments', 'image_set',
                    'obsidianUIMode', 'date_created', 'date_modified'
                ]
                
                # Write known properties in order
                for prop in property_order:
                    if prop in updated_frontmatter:
                        value = updated_frontmatter[prop]
                        
                        # Handle list properties
                        if isinstance(value, list):
                            f.write(f'{prop}:\n')
                            if value:  # Only write items if list has content
                                for item in value:
                                    f.write(f'  - {item}\n')
                        # Handle boolean properties
                        elif isinstance(value, bool):
                            f.write(f'{prop}: {str(value).lower()}\n')
                        # Handle empty string (preserve empty fields like iafd:)
                        elif value == '':
                            f.write(f'{prop}:\n')
                        # Handle other properties
                        else:
                            f.write(f'{prop}: {value}\n')
                
                # Write any custom properties that weren't in our known list
                for prop, value in updated_frontmatter.items():
                    if prop not in property_order:
                        if isinstance(value, list):
                            f.write(f'{prop}:\n')
                            if value:
                                for item in value:
                                    f.write(f'  - {item}\n')
                        elif isinstance(value, bool):
                            f.write(f'{prop}: {str(value).lower()}\n')
                        elif value == '':
                            f.write(f'{prop}:\n')
                        else:
                            f.write(f'{prop}: {value}\n')
                
                f.write('---\n')
                
                # Handle body content
                if old_content:
                    # Extract body (everything after second ---)
                    body = old_content.split('---', 2)[2]
                    
                    # Remove leading newlines and write exactly one blank line
                    body = body.lstrip('\n')
                    f.write('\n')
                    
                    # Find where to insert new videos (before next H3 section after ### Vids)
                    lines = body.split('\n')
                    vids_section_end = -1
                    
                    # Find the ### Vids section
                    for i, line in enumerate(lines):
                        if line.strip().startswith('### Vids'):
                            # Look for the next H3 section after Vids
                            for j in range(i + 1, len(lines)):
                                if lines[j].strip().startswith('###'):
                                    vids_section_end = j
                                    break
                            break
                    
                    # Collect new videos to add
                    new_video_entries = []
                    for video in videos:
                        video_name = os.path.splitext(video['short_filename'])[0]
                        if f'[[{video_name}' in body:
                            continue
                        
                        current_path = video['current_path']
                        file_uri = self.encode_file_uri(current_path)
                        vlc_url = self.encode_network_url(current_path)
                        alias_text = os.path.splitext(video['original_filename'])[0]
                        
                        entry_lines = []
                        if video.get('thumbnail_path'):
                            thumb_filename = os.path.basename(video['thumbnail_path'])
                            entry_lines.append(f'![[{thumb_filename}]]')
                            entry_lines.append(f'[[{video_name}|{alias_text}]]')
                            entry_lines.append('> [!|noicon]')
                            entry_lines.append(f'> [▶ Play Video](<{file_uri}>)')
                            entry_lines.append(f'> [▶ Play Video](<{vlc_url}>)')
                            entry_lines.append('')  # Empty line after entry
                        else:
                            entry_lines.append(f'[[{video_name}|{alias_text}]]')
                            entry_lines.append('> [!|noicon]')
                            entry_lines.append(f'> [▶ Play Video](<{file_uri}>)')
                            entry_lines.append(f'> [▶ Play Video](<{vlc_url}>)')
                            entry_lines.append('')  # Empty line after entry
                        
                        new_video_entries.extend(entry_lines)
                    
                    # Insert new videos at the right position
                    if new_video_entries:
                        if vids_section_end > 0:
                            # Insert before the next H3 section
                            lines.insert(vids_section_end, '')  # Empty line separator
                            for entry_line in reversed(new_video_entries):
                                lines.insert(vids_section_end, entry_line)
                        else:
                            # No H3 section after Vids, append to end
                            lines.extend(new_video_entries)
                    
                    # Write the modified content
                    f.write('\n'.join(lines))
                    # Ensure file ends with newline
                    if lines and lines[-1].strip():
                        f.write('\n')
                else:
                    # New file - write exactly one blank line after ---
                    f.write('\n')
                    f.write(f'# {title_with_descriptors}\n\n')
                    f.write(f'**UUID:** `{actress_uuid}`\n\n')
                    f.write(f'### {actress_name} headshot\n\n')
                    f.write('- place_only_one_pic_here\n\n')
                    f.write('### Extra info\n\n')
                    f.write('- place_iafd_url_where_applicable_here\n\n')
                    f.write('### Vids\n\n')
                    for video in videos:
                        video_name = os.path.splitext(video['short_filename'])[0]
                        current_path = video['current_path']
                        file_uri = self.encode_file_uri(current_path)
                        vlc_url = self.encode_network_url(current_path)
                        alias_text = os.path.splitext(video['original_filename'])[0]
                        if video.get('thumbnail_path'):
                            thumb_filename = os.path.basename(video['thumbnail_path'])
                            f.write(f'![[{thumb_filename}]]\n')
                            f.write(f'[[{video_name}|{alias_text}]]\n')
                            f.write(f'> [!|noicon]\n')
                            f.write(f'> [▶ Play Video](<{file_uri}>)\n')
                            f.write(f'> [▶ Play Video](<{vlc_url}>)\n\n')
                        else:
                            f.write(f'[[{video_name}|{alias_text}]]\n')
                            f.write(f'> [!|noicon]\n')
                            f.write(f'> [▶ Play Video](<{file_uri}>)\n')
                            f.write(f'> [▶ Play Video](<{vlc_url}>)\n\n')
            
            status = "Updated" if old_data else "Created"
            self.log(f"✅ {status} actress note: {actress_uuid}.md", "success")
            
        except Exception as e:
            self.log(f"❌ Error creating actress note: {str(e)}", "error")
    
    def create_video_note(self, video_meta, stash_base):
        try:
            actress_uuid = video_meta['actress_uuid']
            video_name = os.path.splitext(video_meta['short_filename'])[0]
            
            first_char = actress_uuid[0].upper()
            videos_dir = os.path.join(stash_base, 'Videos', first_char)
            os.makedirs(videos_dir, exist_ok=True)
            
            file_path = os.path.join(videos_dir, f"{video_name}.md")
            
            current_path = video_meta['current_path']
            file_uri = self.encode_file_uri(current_path)
            vlc_url = self.encode_network_url(current_path)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write('---\n')
                f.write(f'title: {video_meta["actress_name"]} - {video_meta["movie"] if video_meta["movie"] else video_name}\n')
                f.write(f'original_filename: {video_meta["original_filename"]}\n')
                
                f.write('aliases:\n')
                f.write(f'  - {os.path.splitext(video_meta["original_filename"])[0]}\n')
                
                f.write('tags:\n')
                for tag in sorted(video_meta['tags']):
                    f.write(f'  - {tag}\n')
                
                f.write('dg-publish: false\n')
                f.write(f'madeupname: {str(video_meta["madeupname"]).lower()}\n')
                f.write(f'usualhaircolor: {video_meta["haircolor"]}\n')
                f.write(f'bodytype: {video_meta["bodytype"]}\n')
                f.write(f'nationality: {video_meta["nationality"]}\n')
                f.write(f'actressactive_when: {video_meta["actressactive_when"]}\n')
                
                f.write('partners_with:\n')
                for partner in sorted(video_meta['partners']):
                    f.write(f'  - {partner}\n')
                
                f.write(f'bbc: {str(video_meta["has_bbc"]).lower()}\n')
                f.write(f'anim: {str(video_meta["has_anim"]).lower()}\n')
                f.write(f'aie: {str(video_meta.get("aie", False)).lower()}\n')
                f.write(f'captions: {str(video_meta.get("has_captions", False)).lower()}\n')
                f.write(f'rating: {video_meta["rating_stars"]}\n')
                
                f.write(f'movie: {video_meta["movie"]}\n')
                if video_meta['movie_alt']:
                    f.write('movie_alt:\n')
                    for alt_title in video_meta['movie_alt']:
                        f.write(f'  - {alt_title}\n')
                if video_meta.get('version'):
                    f.write(f'version: "{video_meta["version"]}"\n')
                
                f.write(f'actress: "[[{actress_uuid}]]"\n')
                f.write(f'sequence: {video_meta["sequence_number"]}\n')
                
                if video_meta.get('card_path'):
                    card_filename = os.path.basename(video_meta['card_path'])
                    f.write(f'thumbnail: "[[{card_filename}]]"\n')
                elif video_meta.get('thumbnail_path'):
                    thumb_filename = os.path.basename(video_meta['thumbnail_path'])
                    f.write(f'thumbnail: "[[{thumb_filename}]]"\n')
                
                f.write(f'path: {file_uri}\n')
                f.write(f'path_ipad: {vlc_url}\n')
                # Technical metadata
                if video_meta.get('format'):
                    f.write(f'format: {video_meta["format"]}\n')
                if video_meta.get('file_size_mb') is not None:
                    f.write(f'file_size_mb: {video_meta["file_size_mb"]}\n')
                if video_meta.get('duration'):
                    f.write(f'duration: {video_meta["duration"]}\n')
                if video_meta.get('bitrate'):
                    f.write(f'bitrate: {video_meta["bitrate"]}\n')
                if video_meta.get('width'):
                    f.write(f'width: {video_meta["width"]}\n')
                if video_meta.get('height'):
                    f.write(f'height: {video_meta["height"]}\n')
                if video_meta.get('aspect_ratio'):
                    f.write(f'aspect_ratio: {video_meta["aspect_ratio"]}\n')
                if video_meta.get('frame_rate'):
                    f.write(f'frame_rate: {video_meta["frame_rate"]}\n')
                if video_meta.get('video_codec'):
                    f.write(f'video_codec: {video_meta["video_codec"]}\n')
                if video_meta.get('audio_codec'):
                    f.write(f'audio_codec: {video_meta["audio_codec"]}\n')
                f.write('obsidianUIMode: preview\n')
                f.write(f'date_created: {datetime.now().strftime("%Y-%m-%dT%H:%M")}\n')
                f.write(f'date_modified: {datetime.now().strftime("%Y-%m-%dT%H:%M")}\n')
                f.write('---\n\n')
                
                f.write(f'# {video_meta["actress_name"]} - {video_meta["movie"] if video_meta["movie"] else video_name}\n\n')
                
                f.write(f'**Actress:** [[{actress_uuid}|{video_meta["actress_name"]}]]\n')
                if video_meta['rating_stars']:
                    f.write(f'**Rating:** {video_meta["rating_stars"]}\n')
                if video_meta['movie']:
                    f.write(f'**Movie:** {video_meta["movie"]}\n')
                    if video_meta['movie_alt']:
                        # Format alt titles as comma-separated string
                        alt_titles_str = ', '.join(video_meta['movie_alt'])
                        f.write(f'**Alt Title:** {alt_titles_str}\n')
                    if video_meta.get('version'):
                        f.write(f'**Version:** {video_meta["version"]}\n')
                f.write('\n')
                
                f.write('### Description\n\n')
                f.write(f"{os.path.splitext(video_meta['original_filename'])[0]}\n\n")
                
                f.write('### Additional info\n\n')
                if video_meta.get('file_comments'):
                    f.write(f"**Comments:** {video_meta['file_comments']}\n\n")
                
                if video_meta.get('bracket_info'):
                    f.write('**Notes from filename:**\n')
                    for note in video_meta['bracket_info']:
                        f.write(f'- {note}\n')
                    f.write('\n')
                
                if video_meta['timestamps']:
                    f.write('**Timestamps:**\n')
                    for time, note in sorted(video_meta['timestamps'].items()):
                        f.write(f'- {time}: {note}\n')
                    f.write('\n')
                
                if video_meta['partners']:
                    f.write(f'**Partners:** {", ".join(video_meta["partners"])}\n\n')
                
                f.write('### Vid\n\n')
                if video_meta.get('thumbnail_path'):
                    thumb_filename = os.path.basename(video_meta['thumbnail_path'])
                    f.write(f'![[{thumb_filename}]]\n')
                f.write(f'> [!|noicon]\n')
                f.write(f'> [▶ Play Video](<{file_uri}>)\n')
                f.write(f'> [▶ Play Video](<{vlc_url}>)\n')
            
            self.log(f"✅ Created video note: {video_name}.md", "success")
            
        except Exception as e:
            self.log(f"❌ Error creating video note: {str(e)}", "error")

    def reprocess_thumbnails(self):
        stash_base = self.stash_base.get()
        files = []
        if self.selected_video_files:
            files = [f for f in self.selected_video_files if os.path.exists(f)]
        elif self.single_video_file.get():
            parts = [p.strip() for p in self.single_video_file.get().split(';') if p.strip()]
            files = [p for p in parts if os.path.exists(p)]
        else:
            self.log("⚠️ No files selected for reprocessing", "warning")
            return
        for f in files:
            try:
                self._reprocess_single_thumbnail(f, stash_base)
            except Exception as e:
                self.log(f"⚠️ Reprocess failed: {str(e)}", "warning")

    def _reprocess_single_thumbnail(self, video_path, stash_base):
        base = os.path.splitext(os.path.basename(video_path))[0]
        m = re.match(r'^([A-Za-z][a-z]{2}[A-Z0-9]{5})_(\d{3})_?', base)
        if not m:
            self.log(f"⚠️ Not a UUID-renamed video: {base}", "warning")
            return
        actress_uuid = m.group(1)
        first_char = actress_uuid[0].upper()
        resources_dir = os.path.join(stash_base, first_char, '_resources')
        os.makedirs(resources_dir, exist_ok=True)
        videos_resources_dir = os.path.join(stash_base, 'Videos', first_char, '_resources')
        os.makedirs(videos_resources_dir, exist_ok=True)
        thumb_path = os.path.join(resources_dir, f"{base}_thumb.jpg")
        card_path = os.path.join(videos_resources_dir, f"{base}_card.jpg")
        ok_thumb = self.generate_contact_sheet(video_path, thumb_path)
        ok_card = self.generate_card_thumbnail(video_path, card_path)
        chosen = None
        if ok_card:
            chosen = os.path.basename(card_path)
        elif ok_thumb:
            chosen = os.path.basename(thumb_path)
        file_path = os.path.join(stash_base, 'Videos', first_char, f"{base}.md")
        if not os.path.exists(file_path):
            self.log(f"⚠️ Video note not found: {file_path}", "warning")
            return
        old_data, old_content = self.read_existing_markdown(file_path)
        if not old_content:
            self.log(f"⚠️ Cannot read note: {file_path}", "warning")
            return
        updated_frontmatter = old_data.copy() if old_data else {}
        if chosen:
            updated_frontmatter['thumbnail'] = f'[[{chosen}]]'
        property_order = [
            'title', 'original_filename', 'aliases', 'tags', 'dg-publish',
            'madeupname', 'usualhaircolor', 'bodytype', 'nationality',
            'actressactive_when', 'partners_with', 'bbc', 'anim', 'aie',
            'captions', 'rating', 'movie', 'movie_alt', 'version', 'actress',
            'sequence', 'thumbnail', 'path', 'path_ipad', 'format',
            'file_size_mb', 'duration', 'bitrate', 'width', 'height',
            'aspect_ratio', 'frame_rate', 'video_codec', 'audio_codec',
            'obsidianUIMode', 'date_created', 'date_modified'
        ]
        body = old_content.split('---', 2)[2]
        if chosen:
            lines = body.split('\n')
            callout_idx = None
            for i, line in enumerate(lines):
                if line.strip().startswith('> [!|noicon]'):
                    callout_idx = i
                    break
            if callout_idx is not None:
                prev_line = lines[callout_idx - 1].strip() if callout_idx > 0 else ''
                if not re.match(r'!\[\[', prev_line):
                    lines.insert(callout_idx, f'![[{chosen}]]')
                    body = '\n'.join(lines)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('---\n')
            for prop in property_order:
                if prop in updated_frontmatter:
                    value = updated_frontmatter[prop]
                    if isinstance(value, list):
                        f.write(f'{prop}:\n')
                        if value:
                            for item in value:
                                f.write(f'  - {item}\n')
                    elif isinstance(value, bool):
                        f.write(f'{prop}: {str(value).lower()}\n')
                    elif value == '':
                        f.write(f'{prop}:\n')
                    else:
                        f.write(f'{prop}: {value}\n')
            for prop, value in updated_frontmatter.items():
                if prop not in property_order:
                    if isinstance(value, list):
                        f.write(f'{prop}:\n')
                        if value:
                            for item in value:
                                f.write(f'  - {item}\n')
                    elif isinstance(value, bool):
                        f.write(f'{prop}: {str(value).lower()}\n')
                    elif value == '':
                        f.write(f'{prop}:\n')
                    else:
                        f.write(f'{prop}: {value}\n')
            f.write('---\n')
            f.write(body.lstrip('\n'))
        actress_path = os.path.join(stash_base, first_char, f"{actress_uuid}.md")
        if os.path.exists(actress_path) and chosen:
            try:
                with open(actress_path, 'r', encoding='utf-8') as af:
                    acontent = af.read()
                alines = acontent.split('\n')
                target_idx = None
                for i, line in enumerate(alines):
                    s = line.strip()
                    if s.startswith(f'[[{base}|') or s.startswith(f'- [[{base}|'):
                        target_idx = i
                        break
                if target_idx is not None:
                    callout_idx = None
                    for j in range(target_idx + 1, min(target_idx + 8, len(alines))):
                        if alines[j].strip().startswith('> [!|noicon]'):
                            callout_idx = j
                            break
                    if callout_idx is not None:
                        prev_line = alines[callout_idx - 1].strip() if callout_idx > 0 else ''
                        if not re.match(r'!\[\[', prev_line):
                            alines.insert(callout_idx, f'![[{chosen}]]')
                            with open(actress_path, 'w', encoding='utf-8') as af:
                                af.write('\n'.join(alines))
            except Exception as e:
                self.log(f"  ⚠️ Actress note embed failed: {str(e)}", "warning")
        self.log(f"✅ Reprocessed thumbnails for: {base}", "success")

def main():
    root = tk.Tk()
    app = VideoProcessorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
