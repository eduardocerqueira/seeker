#date: 2023-05-12T16:44:37Z
#url: https://api.github.com/gists/f8f30e5d81e5a24694e74629bb4638d1
#owner: https://api.github.com/users/SameerIndrayan

import PySimpleGUI as sg
import time

class Timer:
    def __init__(self):
        self.study_time = 0
        self.break_time = 0
        self.start_time = 0
        self.end_time = 0
        self.window = None

    def run(self):
        # gui layout (buttons and text)
        layout = [
            [sg.Text('Study Timer')],
            [sg.Text('Study time (minutes):'), sg.Input(key='study_time')],
            [sg.Text('Break time (minutes):'), sg.Input(key='break_time')],
            [sg.Button('START')],
            [sg.Button('CANCEL')],
            [sg.Text('Time remaining:')],
            [sg.Text(size=(10, 2), key='timer')],
        ]

        self.window = sg.Window('Timer', layout)

        while True:
            event, values = self.window.read()

            # if the user closes the window or clicks "CANCEL"
            if event == sg.WINDOW_CLOSED or event == 'CANCEL':
                break

            # if the user clicks "START"
            if event == 'START':
                # get study and break times from the inputs
                self.study_time = int(values['study_time']) * 60
                self.break_time = int(values['break_time']) * 60

                # start the study timer countdown
                self.start_time = time.time()
                self.end_time = self.start_time + self.study_time

                while True:
                    if event == sg.WINDOW_CLOSED or event == 'CANCEL':
                        break
                    current_time = time.time()
                    remaining_time = round(self.end_time - current_time)

                    # if study timer is over
                    if remaining_time <= 0:
                        sg.Popup('Study time is over. Take a break!', title='Time is up!')
                        break

                    # update the countdown in the gui
                    minutes, seconds = divmod(remaining_time, 60)
                    self.window['timer'].update('{:02d}:{:02d}'.format(minutes, seconds))
                    self.window.refresh() # refresh gui every 0.1 sec

                    time.sleep(0.1)

                # start the break timer countdown
                self.start_time = time.time()
                self.end_time = self.start_time + self.break_time

                while True:
                    current_time = time.time()
                    remaining_time = round(self.end_time - current_time)

                    # text when break timer has finished
                    if remaining_time <= 0:
                        sg.Popup('Break time is over. Time to study again!', title='Time is up!')
                        break

                    # update the timer every second in the gui
                    minutes, seconds = divmod(remaining_time, 60)
                    self.window['timer'].update('{:02d}:{:02d}'.format(minutes, seconds))
                    self.window.refresh() # update gui very often

                    time.sleep(0.1)


        self.window.close()

