#date: 2022-01-17T17:03:31Z
#url: https://api.github.com/gists/65d4f3566407e58ced2cf75b7144964b
#owner: https://api.github.com/users/johnlarkin1

"""Send an email alert if the desired process cannot be found."""
import os

from typing import List

import psutil

from base64 import b64decode
from dotenv import load_dotenv
from utils.email_sender import EmailSender

load_dotenv()

# Gmail Integration
GMAIL_SENDER_ADDRESS = os.environ.get("GMAIL_SENDER_ADDRESS")
GMAIL_SENDER_PASSWORD = os.environ.get("GMAIL_SENDER_PASSWORD")


class ProcessSearcher:
    def __init__(self):
        self._email_sender = EmailSender(GMAIL_SENDER_ADDRESS, GMAIL_SENDER_PASSWORD)

    def find_process_by_name_and_alert(self, target_name):
        is_found = False
        for proc in psutil.process_iter():
            try:
                pinfo = proc.as_dict(attrs=['pid', 'name', 'create_time', 'cmdline'])
                # We only want to look at python processes
                if 'python' in pinfo['name'].lower():
                    # We want to look for the specific application executing and make
                    # sure that it's still up
                    cmd : List[str] = pinfo['cmdline']
                    executing_cmd = ' '.join(cmd)
                    if target_name in executing_cmd:
                        print('Found target name in executing command. Sending information.')
                        is_found = True

            except (psutil.NoSuchProcess, psutil.AccessDenied , psutil.ZombieProcess):
                pass

        desired_email = 'your-desired-email@gmail.com'
        if not is_found:
            # TODO(@larkin): Restart automatically and alert
            print(f'Scanned through all running processes and did not find {target_name}')
            print(f'Sending {desired_email} an alert.')
            self._email_sender.send_email_htmltext(
                desired_email,
                f'🆘⚠️ Did not find {target_name} process running. ⚠️🆘',
                'Please check the EC2 instance and restart service if need be.'
            )
        else:
            self._email_sender.send_email_htmltext(
                desired_email,
                f'✅🚀 Service {target_name} is up and running. 🚀✅',
                'All good!! Keep juicing.'
            )

if __name__ == '__main__':
    searcher = ProcessSearcher()
    searcher.find_process_by_name_and_alert('app.py')