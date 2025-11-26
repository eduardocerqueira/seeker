#date: 2025-11-26T17:10:13Z
#url: https://api.github.com/gists/c29bd92692d97df9f48754276d54bf45
#owner: https://api.github.com/users/rangeldarosa

import time
import pyautogui

# Desativa o "failsafe" padrão do pyautogui (mover o mouse para o canto superior esquerdo)
# Se você quiser manter a proteção, comente essa linha.
pyautogui.FAILSAFE = False

try:
    print("Pressione Ctrl+C para parar o script.")
    while True:
        # Pressiona Scroll Lock duas vezes para não deixar o LED diferente do estado original
        # (na prática, qualquer tecla funciona para simular atividade)
        pyautogui.press('scrolllock')
        pyautogui.press('scrolllock')

        # Aguarda 30 segundos
        time.sleep(30)

except KeyboardInterrupt:
    print("\nScript parado pelo usuário.")
