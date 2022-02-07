#date: 2022-02-07T17:09:04Z
#url: https://api.github.com/gists/28f0cf41214a486c0d0192162321e963
#owner: https://api.github.com/users/horstjens

import PySimpleGUI as sg

sg.theme('BluePurple')

layout = [[sg.Text('Your typed chars appear here:'), sg.Text(size=(15,1), key='-OUTPUT-')],
          [sg.Input(key='-IN-')],
          [sg.Button('Show'), sg.Button('Exit')]]

window = sg.Window('Pattern 2B', layout)

while True:  # Event Loop
    event, values = window.read()
    print(event, values)
    if event == sg.WIN_CLOSED or event == 'Exit':
        break
    if event == sg.TIMEOUT
    if event == 'Show':
        # Update the "output" text element to be the value of "input" element
        window['-OUTPUT-'].update(values['-IN-'])

window.close()