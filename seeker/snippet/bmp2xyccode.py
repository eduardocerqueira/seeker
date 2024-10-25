#date: 2024-10-25T14:45:38Z
#url: https://api.github.com/gists/c79ba2a9038c7792c9072635ef0af24c
#owner: https://api.github.com/users/NapoliN

'''
    bmpファイルを読み込んで、RGB値を x,y,#ccode,...の形式で書きだす
'''
import cv2
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
# 読み込むファイル
parser.add_argument('file', type=str, help='file name')
# 出力先 -o か --output で指定
parser.add_argument('-o', '--output', type=str, help='output file name')
args = parser.parse_args()

if args.file is None:
    print("Please specify the file name")
    exit()

# BMP画像の読み込み
img = cv2.imread(args.file)
if img is None:
    print("File not found")
    exit()

# 画像のサイズ
height, width, channels = img.shape

def convertRGB(arr):
    r = format(arr[2],"02x")
    g = format(arr[1],"02x")
    b = format(arr[0],"02x")
    return "#" + r +  g + b

with open(args.output,"w") as f:
    tmp = ""
    for y,col in enumerate(img):
        for x, arr in enumerate(col):
                if not np.all(arr==0):
                    tmp += f"{x},{y},{convertRGB(arr)},"
    f.write(tmp[:-1])