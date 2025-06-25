#date: 2025-06-25T17:01:23Z
#url: https://api.github.com/gists/e17f28da1ac0924f0841a8406c52b111
#owner: https://api.github.com/users/kazuki0824

import json
from deepdiff import DeepDiff
from deepdiff.helper import CannotCompare

def compare_by_parts_id(x, y, level=None):
    """
    DeepDiff の iterable_compare_func 用コールバック:
    各要素の abstract.PartsID をキーに対応付ける
    """
    try:
        # dict として扱う場合
        tag_x = x.get('abstract', {}).get('PartsID')
        tag_y = y.get('abstract', {}).get('PartsID')
        return tag_x == tag_y
    except Exception:
        # フォールバックしてインデックス順比較に戻す
        raise CannotCompare()

def load_json(path):
    """ファイルパスを受け取り JSON を読み込んで返す"""
    with open(path, encoding='utf-8') as f:
        return json.load(f)

if __name__ == '__main__':
    # 比較対象ファイルのパス
    file1 = 'object1.json'
    file2 = 'object2.json'

    # ファイルから読み込み
    obj1 = load_json(file1)
    obj2 = load_json(file2)

    # DeepDiff 実行
    diff = DeepDiff(
        obj1,
        obj2,
        ignore_order=True,              # リストの順序は無視
        iterable_compare_func=compare_by_parts_id,
        verbose_level=2
    )

    # 結果を出力
    print('--- Differences ---')
    print(diff)
