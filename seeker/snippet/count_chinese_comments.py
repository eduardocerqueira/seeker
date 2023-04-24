#date: 2023-04-24T16:51:57Z
#url: https://api.github.com/gists/732e31624d0f02fd928389ac70cdc5f0
#owner: https://api.github.com/users/goocarlos

import os
import re
import time

from colorama import Fore, Style

"""
该程序可以找到所有项目中包含所有 Python 文件中的中文注释
This program can find all Chinese comments in all Python files in the project
Author: goocarlos<goocarlos@gmail.com>
License: MIT
"""

# 修改为你的项目根目录
root_directory = "path/to/your/project/root/directory"

# 正则表达式匹配中文字符
chinese_pattern = re.compile(r'[\u4e00-\u9fff]+')


def find_chinese_comments_in_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    count = 0
    for line in lines:
        if chinese_pattern.search(line):
            count += 1

    return count


start_time = time.time()
files = []
comments = []
for root, _, filenames in os.walk(root_directory):
    for filename in filenames:
        if filename.endswith('.py'):
            file_path = os.path.join(root, filename)
            relpath = os.path.relpath(file_path, root_directory)
            count = find_chinese_comments_in_file(file_path)
            if count > 0:
                files.append(relpath)
                comments.append(count)

end_time = time.time()
file_header = f'{Fore.LIGHTMAGENTA_EX}File{Fore.LIGHTBLACK_EX}'
comments_header = f'{Fore.LIGHTYELLOW_EX}Comments{Fore.LIGHTBLACK_EX}'
print(f"{file_header:<30} {comments_header:>10}")
print('=' * 45)
for file, count in zip(files, comments):
    file_colored = f'{Fore.LIGHTCYAN_EX}{file}{Fore.LIGHTBLACK_EX}'
    count_colored = f'{Fore.LIGHTGREEN_EX}{count}{Fore.LIGHTBLACK_EX}'
    print(f"{file_colored:<30} {count_colored:>10}{Style.RESET_ALL} comments")

print(f'\n{Fore.LIGHTYELLOW_EX}Running time:{Fore.LIGHTBLACK_EX} ',
      f'{end_time - start_time:.2f}s{Style.RESET_ALL}')
