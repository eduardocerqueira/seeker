#date: 2024-08-20T16:46:10Z
#url: https://api.github.com/gists/e418fe42d33f953ac4a0d0cea933b18d
#owner: https://api.github.com/users/todashuta

import os
import re
import glob
import time
from osgeo import (gdal, osr)
import numpy as np


"""
lem ファイルを GeoTIFF に変換するスクリプトです。

lem ファイルの仕様
https://www.gsi.go.jp/MAP/CD-ROM/dem5m/doc/info5m1.htm


（現時点では） lem2gtif.py が存在するフォルダ内の lem ファイルを
GeoTIFF に変換するようになっているので、このファイルを対象の
フォルダにコピーし、実行してください。
"""



def main():
    # lem2gtif.py と同じフォルダ内に存在する lem ファイルを走査
    basedir = os.path.dirname(os.path.abspath(__file__))
    for lem_filepath in glob.glob(os.path.join(basedir, '*.lem')):
        try:
            # lem ファイルを GeoTIFF に変換
            lem_to_gtif(lem_filepath)
        except (ValueError, FileNotFoundError) as err:
            print(f"Error: {err}")
            print()



def lem_to_gtif(lem_filepath):
    if not re.search(r'\.lem$', lem_filepath):
        raise ValueError("lem ファイル以外が指定されました。")

    # lem ファイルと同じ場所に csv 形式のヘッダファイルがあるはず
    csv_filepath = re.sub(r'\.lem$', '.csv', lem_filepath)
    # lem ファイルと同じ場所に変換した TIFF ファイルを出力する
    tif_filepath = re.sub(r'\.lem$', '.tif', lem_filepath)

    lem_filename = os.path.basename(lem_filepath)
    csv_filename = os.path.basename(csv_filepath)
    tif_filename = os.path.basename(tif_filepath)

    now = time.strftime('%H:%M:%S', time.localtime())
    print(f"[{now}] convert to {tif_filename}")

    # ファイルが存在しているかチェック
    if not os.path.exists(lem_filepath):
        raise FileNotFoundError(f"{lem_filename} がみつかりません。")
    if not os.path.exists(csv_filepath):
        raise FileNotFoundError(f"{csv_filename} がみつかりません。")

    # ヘッダファイルを読み取る
    header = read_header_file(csv_filepath)
    if header is None:
        raise ValueError(f"{csv_filename} のパースに失敗しました。")

    # GeoTIFF ファイル作成
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(header['crs'])
    driver = gdal.GetDriverByName('GTiff')
    raster = driver.Create(tif_filepath, header['cols'], header['rows'], 1, gdal.GDT_Float32)
    raster.SetGeoTransform((header['left'], header['xres'], 0, header['top'], 0, -header['yres']))
    raster.SetProjection(srs.ExportToWkt())
    band = raster.GetRasterBand(1)
    band.SetNoDataValue(-9999.0)

    with open(lem_filepath) as lem_f:
        row = 0
        # lem ファイルから1行読み取り、 GeoTIFF ファイルに1行書き込む
        for line in lem_f:
            line = line.replace('-1111', '-9999')
            array = np.array([[-9999.0 if line[5*i+10:5*i+15] == '-9999' else int(line[5*i+10:5*i+15])/10 for i in range(header['cols'])]])
            band.WriteArray(array, yoff=row)
            row += 1
        band.FlushCache()



def read_header_file(csv_filepath):
    csv_filename = os.path.basename(csv_filepath)
    header = {}

    # ヘッダファイルを読み込む
    with open(csv_filepath, 'r', encoding='cp932') as f:
        for line in f:
            k, v = line.strip().split(',')
            if k == '東西方向の点数':
                header['cols'] = int(v)
            elif k == '南北方向の点数':
                header['rows'] = int(v)
            elif k == '東西方向のデータ間隔':
                header['xres'] = float(v)
            elif k == '南北方向のデータ間隔':
                header['yres'] = float(v)
            elif k == '平面直角座標系番号' or k == '座標系番号':
                header['crs'] = int(v) + 6668
            elif k == '区画左下X座標' or k == '区画左下Ｘ座標':
                header['bottom'] = int(v) / 100
            elif k == '区画左下Y座標' or k == '区画左下Ｙ座標':
                header['left'] = int(v) / 100
            elif k == '区画右上X座標' or k == '区画右上Ｘ座標':
                header['top'] = int(v) / 100
            elif k == '区画右上Y座標' or k == '区画右上Ｙ座標':
                header['right'] = int(v) / 100

    # 必要な情報が読み取れなかったら例外を投げる
    if 'cols' not in header:
        raise ValueError(f"{csv_filename} に「東西方向の点数」がありません。")
    if 'rows' not in header:
        raise ValueError(f"{csv_filename} に「南北方向の点数」がありません。")
    if 'xres' not in header:
        raise ValueError(f"{csv_filename} に「東西方向のデータ間隔」がありません。")
    if 'yres' not in header:
        raise ValueError(f"{csv_filename} に「南北方向のデータ間隔」がありません。")
    if 'crs' not in header:
        raise ValueError(f"{csv_filename} に「座標系番号」がありません。")
    if 'bottom' not in header:
        raise ValueError(f"{csv_filename} に「区画左下Ｘ座標」がありません。")
    if 'left' not in header:
        raise ValueError(f"{csv_filename} に「区画左下Ｙ座標」がありません。")
    if 'top' not in header:
        raise ValueError(f"{csv_filename} に「区画右上Ｘ座標」がありません。")
    if 'right' not in header:
        raise ValueError(f"{csv_filename} に「区画右上Ｙ座標」がありません。")

    return header



main()