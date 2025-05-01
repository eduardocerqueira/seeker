#date: 2025-05-01T17:11:53Z
#url: https://api.github.com/gists/7b0455e4a7725cae976db910d8de399c
#owner: https://api.github.com/users/changemewtf

from pywinauto.application import Application

import os
import time, datetime
import threading, subprocess

def find_record_dialog(fcadefbneo_path):
    app = Application(backend="win32").connect(path=f"{fcadefbneo_path}\\fcadefbneo.exe")
    win = app.top_window()
    win.menu_select("Game->RecordAVI")
    # Assumes x264vfw codec has been installed to Windows
    app.SetVideoCompressionOption.ComboBox.select(4)
    app.SetVideoCompressionOption.Ok.click()
    return True

def start_fcadefbneo(fcadefbneo_path, challenge_id, game_name):
    command = " ".join([ 
        f'{fcadefbneo_path}\\fcadefbneo.exe',
        f'quark:stream,{game_name},{challenge_id}.2,7100'
    ])
    subprocess.run(command)
    
def main(challenge_id=None, recording_time=None, kill_time=None, fcadefbneo_path=None, game_name=None):
    begin_time = datetime.datetime.now()

    # Make sure 'started.inf' is missing
    if os.path.exists(f"{fcadefbneo_path}/fightcade/started.inf"):
        os.remove(f"{fcadefbneo_path}/fightcade/started.inf")

    # Start ggpofbneo
    print("Starting fcadefbneo thread")
    ggpo_thread = threading.Thread(target=start_fcadefbneo, args=[
                                   fcadefbneo_path, challenge_id, game_name])
    ggpo_thread.start()
    print("Started ggpofbneo")

    # Check to see if fcadefbneo has started playing
    print('Checking to see if replay has started')
    first_frame = False
    while True:
        running_time = (datetime.datetime.now() - begin_time).seconds

        if os.path.exists(f"{fcadefbneo_path}/fightcade/started.inf"):
            if not first_frame:
                first_frame = True
                print("First frame displayed.")

            print('Looking for recording dialog...')
            if find_record_dialog(fcadefbneo_path):
                break

        # Timeout reached, exiting
        if running_time > kill_time:
            print('Match never started, exiting')
            # cleanup_tasks()
            return "FailTimeout"
        time.sleep(0.1)

    begin_time = datetime.datetime.now()
    while True:
        running_time = (datetime.datetime.now() - begin_time).seconds

        # Log what minute we are on
        if (running_time % 60) == 0:
            print(
                f'Minute: {int(running_time/60)} of {int(recording_time/60)}')

        # Finished recording video
        if running_time > recording_time:
            print("Running time has been reached, stopping the recording.")
            app = Application(backend="win32").connect(path=f"{fcadefbneo_path}\\fcadefbneo.exe")
            win = app.top_window()
            win.wrapper_object().close()
            # cleanup_tasks()
            return "Pass"

        # Kill Timeout reached
        if running_time > (running_time + kill_time):
            print("Kill time reached.")
            return "FailTimeout"
        time.sleep(0.2)


if __name__ == "__main__":
    main(
        challenge_id='1744502546647-8439',
        recording_time=int(8*60+54),
        kill_time=int(10),
        fcadefbneo_path=r'E:\Games\Fightcade\emulator\fbneo',
        game_name='sfiii3nr1'
    )
