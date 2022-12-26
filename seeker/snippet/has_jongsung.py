#date: 2022-12-26T17:05:02Z
#url: https://api.github.com/gists/df99ea22126967fd79a90cad69c95bc1
#owner: https://api.github.com/users/hyunwoongko

# -*- coding: utf-8 -*-

"""
has_jongsung(c) returns True if Hangul Syllable, c has jongsung (which means final), False otherwise.
If c is not a Hangul Syllable, it raises NotHangulSyllables exception.
This function does not work correctly if c is not a single Hangul Syllable.  

has_jongsung(c) 는 한글 음절, c가 종성을 포함하고 있다면 True를, 아니라면 False를 반환한다. 
만약, c 가 한글 음절이 아니라면 NotHangulSyllables exception을 반환한다.
c 가 한 음절이 아닌 경우의 동작에 대해서는 보장할 수 없다.
"""
class NotHangulSyllables(Exception):
    pass

def has_jongsung(c):
    code = ord(c)
    if code < 0xAC00 or code > 0xD7A3:
        raise NotHangulSyllables()

    offset = code - 0xAC00
    if offset % 28 == 0:
        return False
    return True

"""
동작 원리는 다음과 같다. 

HangulSyllables 은 유니코드 테이블에서 0xAC00 ~ 0xD7AF를 차지하고 있다. 
하지만, 실제로 의미있는 글자는 0xD7A3 까지이다.

이 때, 한글 각 글자의 코드는 다음 표를 참고하여 계산할 수 있다. 

      0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27
초성 ㄱ ㄲ ㄴ ㄷ ㄸ ㄹ ㅁ ㅂ ㅃ ㅅ ㅆ ㅇ ㅈ ㅉ ㅊ ㅋ ㅌ ㅍ ㅎ 
중성 ㅏ ㅐ ㅑ ㅒ ㅓ ㅔ ㅕ ㅖ ㅗ ㅘ ㅙ ㅚ ㅛ ㅜ ㅝ ㅞ ㅟ ㅠ ㅡ ㅢ ㅣ
종성    ㄱ ㄲ ㄳ ㄴ ㄵ ㄶ ㄷ ㄹ ㄺ ㄻ ㄼ ㄽ ㄾ ㄿ ㅀ ㅁ ㅂ ㅄ ㅅ ㅆ ㅇ ㅈ ㅊ ㅋ ㅌ ㅍ ㅎ

각 글자의 코드는 '0xAC00 + (초성의 index) * 588 + (중성의 index) * 28 + (종성의 index)'과 같다. 
종성의 0번째 index는 받침이 없는 글자를 말한다. 

예를 들어, "한" = 0xAC00 + 18 * 588 + 0 * 28 + 4 = 0xD55C 이 된다. 

이 계산식을 역으로 풀어보면, 주어진 한글 음절이 종성을 포함하는지 아닌지 판단이 가능하다. 
"""