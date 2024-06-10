#date: 2024-06-10T16:49:22Z
#url: https://api.github.com/gists/51425deb11cdff282ee66aa3fc3f3dea
#owner: https://api.github.com/users/juniorpb

import pyautogui
import time

def click_every_30_seconds():
    try:
        direction = 1  # 1 para adicionar, -1 para subtrair
        while True:
            # Obter a posição atual do mouse
            x, y = pyautogui.position()
            # Ajustar a posição do mouse ligeiramente
            x += direction
            y += direction
            # Realizar o clique na nova posição
            print("clicando em x %s e y%s", x, y)
            pyautogui.click(x, y)
            # Alternar a direção para o próximo clique
            direction *= -1
            # Esperar 30 segundos
            time.sleep(3)
    except KeyboardInterrupt:
        print("Script interrompido pelo usuário.")

if __name__ == "__main__":
    click_every_30_seconds()