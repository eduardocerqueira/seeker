#date: 2026-02-09T17:46:06Z
#url: https://api.github.com/gists/cd877898f7d15eca58a6b251ec1f0a16
#owner: https://api.github.com/users/MatthewJakubowski

import os
import datetime

# TRINITY IDENTITY MODULE
# Handles the 'Me.md' Self-Updating Memory

MEMORY_FILE = "Me.md"

def ensure_memory_file():
    if not os.path.exists(MEMORY_FILE):
        print("⚠️  Me.md not found. Creating new genetic code...")
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            f.write("# TRINITY IDENTITY FILE\n\n## KNOWLEDGE LOG:\n")

def update_memory_interactive():
    ensure_memory_file()
    print("\n--- 🧬 IDENTITY UPDATE PROTOCOL ---")
    print("Co chcesz dodać do swojej tożsamości?")
    print("[1] Nowa Umiejętność (Skill)")
    print("[2] Nowy Cel (Objective)")
    print("[3] Ważna Lekcja (Insight)")
    
    choice = input("Wybór > ")
    content = input("Treść wpisu > ")
    
    categories = {"1": "SKILL", "2": "OBJECTIVE", "3": "INSIGHT"}
    category = categories.get(choice, "NOTE")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    
    entry = f"\n- **[{timestamp}] [{category}]**: {content}"
    
    with open(MEMORY_FILE, "a", encoding="utf-8") as f:
        f.write(entry)
    
    print(f"\n✅ [SUCCESS] Wiedza dodana do pliku {MEMORY_FILE}.")
    print("   Twoje 'Personal Intelligence' zostało zaktualizowane.")
