#date: 2024-04-19T16:58:40Z
#url: https://api.github.com/gists/915ac6ef2fcfc7ac584a1cfd815d4c7d
#owner: https://api.github.com/users/godbyte

# Instalar módulo pyinstaller
pip install pyinstaller

# Generar exe desde script
pyinstaller --onefile myscript.py

# Generar exe desde script sin que se muestre la consola
pyinstaller --onefile --windowed myscript.py