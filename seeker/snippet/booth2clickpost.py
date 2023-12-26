#date: 2023-12-26T16:58:45Z
#url: https://api.github.com/gists/d05f869c9f67af6753154cabcc342ca6
#owner: https://api.github.com/users/takashicompany

import csv
import sys
import os

def count_length(s):
    length = 0
    for char in s:
        if ord(char) > 127:
            length += 1  # 全角文字は1とカウント
        else:
            length += 0.5  # 半角文字は0.5とカウント
    return length

def truncate_content(content):
    length = 0
    truncated = ""
    for char in content:
        if count_length(truncated + char) > 15:
            break
        truncated += char
    return truncated

def convert_csv(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', newline='', encoding='shift_jis', errors='replace') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        # ヘッダーの書き込み
        headers = ["お届け先郵便番号", "お届け先氏名", "お届け先敬称", "お届け先住所1行目", "お届け先住所2行目", "お届け先住所3行目", "お届け先住所4行目", "内容品"]
        writer.writerow(headers)

        next(reader)  # 入力ファイルのヘッダーをスキップ
        for row in reader:
            # 必要なデータを抽出
            postal_code, name, prefecture, address1, address2, product_info = row[8], row[12], row[9], row[10], row[11], row[14]

            # 文字の置換
            postal_code = postal_code.replace('\uff0d', '-')
            name = name.replace('\uff0d', '-')
            prefecture = prefecture.replace('\uff0d', '-')
            address1 = address1.replace('\uff0d', '-')
            address2 = address2.replace('\uff0d', '-')
            product_info = product_info.replace('\uff0d', '-')

            # 内容品の抽出
            product_info_parts = product_info.split("/")
            content_full = product_info_parts[2].strip()
            content = truncate_content(content_full)

            # 新しい行を書き込み
            new_row = [postal_code, name, "様", prefecture, address1, address2, "", content]
            writer.writerow(new_row)

    print(f"変換が完了しました。出力ファイル: {output_file}")

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print("Usage: python script.py <input_file> [output_file]")
        sys.exit(1)
    elif len(sys.argv) == 2:
        input_file = sys.argv[1]
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}_clickpost.csv"
    else:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
    
    convert_csv(input_file, output_file)
