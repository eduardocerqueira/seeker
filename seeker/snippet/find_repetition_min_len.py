#date: 2025-04-23T16:36:53Z
#url: https://api.github.com/gists/c15b4757e38362a9ff420f0fd97d0a18
#owner: https://api.github.com/users/Mxoder

import re

def find_repetition_min_len(text, min_length=3):
    """
    使用正则表达式查找字符串中第一次出现的、且长度不小于指定最小值的
    连续重复子序列。

    Args:
        text: 输入的字符串。
        min_length: 重复子序列的最小长度 (默认为 3)。

    Returns:
        如果找到，返回一个元组 (repeating_substring, start_index, length)，
        包含重复的子序列、重复部分 (如 'AAA...') 的起始索引和子序列的长度。
        如果没有找到，返回 None。
    """
    if not isinstance(min_length, int) or min_length < 1:
        # 最小长度至少为1才有意义
        raise ValueError("min_length 必须是一个正整数")

    # 使用 f-string 动态构建正则表达式
    # .{min_length,}? 匹配至少 min_length 个字符，非贪婪
    pattern = rf'(.{{{min_length},}}?)\1'

    # 使用 re.DOTALL 使 '.' 匹配换行符
    match = re.search(pattern, text, re.DOTALL)

    if match:
        substring = match.group(1)          # 获取捕获组1的内容，即重复的单元
        start_index = match.start()         # 获取整个匹配 (例如 'AAAA') 的起始索引
        length = len(substring)             # 重复单元的长度
        return substring, start_index, length
    else:
        return None