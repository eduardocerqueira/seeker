#date: 2023-12-26T16:42:14Z
#url: https://api.github.com/gists/61b8c3d3cfa0112da9ea1e688a7eea4d
#owner: https://api.github.com/users/eliasalbuquerque

# 202312 - Python 3.12.0


import pyautogui

# Criar pasta em um diretório usando o mouse
# 1. abrir explorer - 1c
print('Abrindo diretório...')
pyautogui.click(1100,1049, duration=1)

# 2. abrir home - 1c
pyautogui.click(337,683, duration=1)

# 3. abrir workspace - 2c
pyautogui.doubleClick(1279,424, duration=1)

# 4. abrir python-automation - 2c
pyautogui.doubleClick(570,806, duration=1)

# 5. abrir assets - 2c
pyautogui.doubleClick(569,393, duration=1)

# 6. mover meio do explorer
print('Criando pasta...')
pyautogui.click(980,568, duration=1)

# 7. botao direito mouse
pyautogui.rightClick()

# 8. mover ate 'New'
pyautogui.click(1150,458, duration=1)

# 9. mover ate 'Folder'
pyautogui.click(1335,458, duration=1)

# 10. clicar meio do explorer
pyautogui.click(980,568, duration=1)
print('Pasta criada com sucesso!')
