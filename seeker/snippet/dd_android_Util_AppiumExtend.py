#date: 2021-08-31T03:24:30Z
#url: https://api.github.com/gists/5187cd8f63ab771bd9bc36e06e9831fb
#owner: https://api.github.com/users/caihao6654

#!/usr/bin/env python
# coding=utf-8

import os
import platform
import tempfile
import shutil

from PIL import Image
from PIL import ImageChops

PATH = lambda p: os.path.abspath(p)
TEMP_FILE = PATH(tempfile.gettempdir() + "/temp_screen.png")


class Appium_Extend(object):
    def __init__(self, driver):
        self.driver = driver

    def get_screenshot_by_element(self, element):
        # 先截取整个屏幕，存储至系统临时目录下
        self.driver.get_screenshot_as_file(TEMP_FILE)

        # 获取元素bounds
        location = element.location
        size = element.size
        box = (location["x"], location["y"], location["x"] + size["width"], location["y"] + size["height"])

        # 截取图片
        image = Image.open(TEMP_FILE)
        newImage = image.crop(box)
        newImage.save(TEMP_FILE)

        return self

    def get_screenshot_by_custom_size(self, start_x, start_y, end_x, end_y):
        # 自定义截取范围
        self.driver.get_screenshot_as_file(TEMP_FILE)
        box = (start_x, start_y, end_x, end_y)

        image = Image.open(TEMP_FILE)
        newImage = image.crop(box)
        newImage.save(TEMP_FILE)

        return self

    def write_to_file(self, dirPath, imageName, form="png"):
        # 将截屏文件复制到指定目录下
        if not os.path.isdir(dirPath):
            os.makedirs(dirPath)
        shutil.copyfile(TEMP_FILE, PATH(dirPath + "/" + imageName + "." + form))

    def load_image(self, image_path):
        # 加载目标图片供对比用
        if os.path.isfile(image_path):
            load = Image.open(image_path)
            return load
        else:
            raise Exception("%s is not exist" % image_path)

    # http://testerhome.com/topics/202
    def same_as(self, load_image, percent):
        # # 对比图片，percent值设为0，则100%相似时返回True，设置的值越大，相差越大
        # import math
        # import operator

        image1 = Image.open(TEMP_FILE)
        image2 = load_image

        try:
            diff = ImageChops.difference(image1, image2)

            if diff.getbbox() is None:
                # 图片间没有任何不同则直接退出
                return True
            else:
                return  False
        except ValueError as e:
            text = ("表示图片大小和box对应的宽度不一致，参考API说明：Pastes another image into this image."
                    "The box argument is either a 2-tuple giving the upper left corner, a 4-tuple defining the left, upper, "
                    "right, and lower pixel coordinate, or None (same as (0, 0)). If a 4-tuple is given, the size of the pasted "
                    "image must match the size of the region.使用2纬的box避免上述问题")
            print("【{0}】{1}".format(e, text))