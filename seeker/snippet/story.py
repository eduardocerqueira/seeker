#date: 2024-07-05T17:01:09Z
#url: https://api.github.com/gists/287f127acf8b4d5876a68195ad7aea06
#owner: https://api.github.com/users/zalexit

import datetime
import os
import sys


if __name__ == '__main__':
   file_name = sys.argv[1]


   print(f'{file_name=}')


   only_file_name = len(sys.argv) == 2
   file_and_username = len(sys.argv) == 3
   file_user_and_story = len(sys.argv) == 4
   now = datetime.datetime.now()


   if os.path.isfile(file_name):
       print('file already exists remove it first.')
       sys.exit(1)


   if only_file_name:
       text = f'file was created by Python script at {now}'
   elif file_and_username:
       username = sys.argv[2]
       text = f'Hi, this is {username}. I was here at {now}'
   elif file_user_and_story:
       username = sys.argv[2]
       story = sys.argv[3]
       text = f'Hi, my name is {username}.\r\n{story}\r\n{now}'
   else:
       text = 'wrong data'


   with open(file_name, 'wt') as txt_file:
       txt_file.write(text)


   abs_path = os.path.abspath(file_name)
   print(f'file created: {abs_path}')