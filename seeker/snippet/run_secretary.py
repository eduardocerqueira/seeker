#date: 2022-02-15T16:57:17Z
#url: https://api.github.com/gists/6d74284a48d4b1352dc15683c394c91d
#owner: https://api.github.com/users/withtwoemms

from datetime import datetime
from sys import exit

from actionpack.actions import ReadInput
from actionpack.actions import Pipeline
from actionpack.actions import Write


filename = 'fake.chat'
listen = ReadInput('What should I record? ')
record = Pipeline.Fitting(
    action=Write,
    **{
        'prefix': f'[{datetime.now()}] ',
        'append': True,
        'filename': filename,
        'to_write': Pipeline.Receiver
    },
)


secretary = Pipeline(listen, record)

if __name__ == '__main__':
    while True:
        try:
            secretary.perform()
        except KeyboardInterrupt:
            print('\ngoodbye.')
            exit(0)
