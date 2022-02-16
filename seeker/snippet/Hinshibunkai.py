#date: 2022-02-16T16:53:02Z
#url: https://api.github.com/gists/90347463a6ddd3dc9ad8cfc49c71e9b8
#owner: https://api.github.com/users/window794

import MeCab
import ipadic
import collections

#CHASEN出力フォーマットの定義
CHASEN_ARGS = r' -F "%m\t%f[7]\t%f[6]\t%F-[0,1,2,3]\t%f[4]\t%f[5]\n"'
CHASEN_ARGS += r' -U "%m\t%m\t%m\t%F-[0,1,2,3]\t\t\n"'
tagger = MeCab.Tagger(ipadic.MECAB_ARGS + CHASEN_ARGS)

print(tagger.parse("図書館にいた事がバレた"))