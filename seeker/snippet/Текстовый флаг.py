#date: 2023-09-22T17:08:37Z
#url: https://api.github.com/gists/425f47d683f770b457e872b1b4a44d55
#owner: https://api.github.com/users/AspirantDrago

import io
import sys

from PyQt5 import uic, QtCore, QtWidgets  # Импортируем uic
from PyQt5.QtWidgets import QApplication, QMainWindow

template = """<?xml version="1.0" encoding="UTF-8"?>
<ui version="4.0">
 <class>MainWindow</class>
 <widget class="QMainWindow" name="MainWindow">
  <property name="geometry">
   <rect>
    <x>0</x>
    <y>0</y>
    <width>575</width>
    <height>404</height>
   </rect>
  </property>
  <property name="windowTitle">
   <string>Текстовый флаг</string>
  </property>
  <widget class="QWidget" name="centralwidget">
   <layout class="QGridLayout" name="gridLayout_2">
    <item row="0" column="0">
     <layout class="QGridLayout" name="gridLayout">
      <item row="0" column="0">
       <widget class="QLabel" name="label">
        <property name="font">
         <font>
          <pointsize>16</pointsize>
          <weight>75</weight>
          <bold>true</bold>
         </font>
        </property>
        <property name="text">
         <string>Цвет № 1</string>
        </property>
        <property name="alignment">
         <set>Qt::AlignCenter</set>
        </property>
       </widget>
      </item>
      <item row="3" column="0">
       <widget class="QRadioButton" name="radioButton_3">
        <property name="font">
         <font>
          <pointsize>12</pointsize>
         </font>
        </property>
        <property name="text">
         <string>Зелёный</string>
        </property>
        <attribute name="buttonGroup">
         <string notr="true">color_group_1</string>
        </attribute>
       </widget>
      </item>
      <item row="1" column="0">
       <widget class="QRadioButton" name="radioButton">
        <property name="font">
         <font>
          <pointsize>12</pointsize>
         </font>
        </property>
        <property name="text">
         <string>Синий</string>
        </property>
        <property name="checked">
         <bool>true</bool>
        </property>
        <attribute name="buttonGroup">
         <string notr="true">color_group_1</string>
        </attribute>
       </widget>
      </item>
      <item row="1" column="1">
       <widget class="QRadioButton" name="radioButton_4">
        <property name="font">
         <font>
          <pointsize>12</pointsize>
         </font>
        </property>
        <property name="text">
         <string>Синий</string>
        </property>
        <property name="checked">
         <bool>true</bool>
        </property>
        <attribute name="buttonGroup">
         <string notr="true">color_group_2</string>
        </attribute>
       </widget>
      </item>
      <item row="3" column="1">
       <widget class="QRadioButton" name="radioButton_6">
        <property name="font">
         <font>
          <pointsize>12</pointsize>
         </font>
        </property>
        <property name="text">
         <string>Зелёный</string>
        </property>
        <attribute name="buttonGroup">
         <string notr="true">color_group_2</string>
        </attribute>
       </widget>
      </item>
      <item row="2" column="0">
       <widget class="QRadioButton" name="radioButton_2">
        <property name="font">
         <font>
          <pointsize>12</pointsize>
         </font>
        </property>
        <property name="text">
         <string>Красный</string>
        </property>
        <attribute name="buttonGroup">
         <string notr="true">color_group_1</string>
        </attribute>
       </widget>
      </item>
      <item row="1" column="2">
       <widget class="QRadioButton" name="radioButton_7">
        <property name="font">
         <font>
          <pointsize>12</pointsize>
         </font>
        </property>
        <property name="text">
         <string>Синий</string>
        </property>
        <property name="checked">
         <bool>true</bool>
        </property>
        <attribute name="buttonGroup">
         <string notr="true">color_group_3</string>
        </attribute>
       </widget>
      </item>
      <item row="0" column="2">
       <widget class="QLabel" name="label_3">
        <property name="font">
         <font>
          <pointsize>16</pointsize>
          <weight>75</weight>
          <bold>true</bold>
         </font>
        </property>
        <property name="text">
         <string>Цвет № 3</string>
        </property>
        <property name="alignment">
         <set>Qt::AlignCenter</set>
        </property>
       </widget>
      </item>
      <item row="2" column="2">
       <widget class="QRadioButton" name="radioButton_8">
        <property name="font">
         <font>
          <pointsize>12</pointsize>
         </font>
        </property>
        <property name="text">
         <string>Красный</string>
        </property>
        <attribute name="buttonGroup">
         <string notr="true">color_group_3</string>
        </attribute>
       </widget>
      </item>
      <item row="4" column="0" colspan="3">
       <widget class="QPushButton" name="make_flag">
        <property name="font">
         <font>
          <pointsize>14</pointsize>
         </font>
        </property>
        <property name="text">
         <string>Сделать флаг</string>
        </property>
       </widget>
      </item>
      <item row="0" column="1">
       <widget class="QLabel" name="label_2">
        <property name="font">
         <font>
          <pointsize>16</pointsize>
          <weight>75</weight>
          <bold>true</bold>
         </font>
        </property>
        <property name="text">
         <string>Цвет № 2</string>
        </property>
        <property name="alignment">
         <set>Qt::AlignCenter</set>
        </property>
       </widget>
      </item>
      <item row="3" column="2">
       <widget class="QRadioButton" name="radioButton_9">
        <property name="font">
         <font>
          <pointsize>12</pointsize>
         </font>
        </property>
        <property name="text">
         <string>Зелёный</string>
        </property>
        <attribute name="buttonGroup">
         <string notr="true">color_group_3</string>
        </attribute>
       </widget>
      </item>
      <item row="2" column="1">
       <widget class="QRadioButton" name="radioButton_5">
        <property name="font">
         <font>
          <pointsize>12</pointsize>
         </font>
        </property>
        <property name="text">
         <string>Красный</string>
        </property>
        <attribute name="buttonGroup">
         <string notr="true">color_group_2</string>
        </attribute>
       </widget>
      </item>
      <item row="5" column="0" colspan="3">
       <widget class="QLabel" name="result">
        <property name="font">
         <font>
          <pointsize>14</pointsize>
         </font>
        </property>
        <property name="text">
         <string/>
        </property>
       </widget>
      </item>
     </layout>
    </item>
   </layout>
  </widget>
  <widget class="QMenuBar" name="menubar">
   <property name="geometry">
    <rect>
     <x>0</x>
     <y>0</y>
     <width>575</width>
     <height>26</height>
    </rect>
   </property>
  </widget>
  <widget class="QStatusBar" name="statusbar"/>
 </widget>
 <resources/>
 <connections/>
 <buttongroups>
  <buttongroup name="color_group_1"/>
  <buttongroup name="color_group_2"/>
  <buttongroup name="color_group_3"/>
 </buttongroups>
</ui>
"""


class FlagMaker(QMainWindow):
    def __init__(self):
        super().__init__()
        f = io.StringIO(template)
        uic.loadUi(f, self)  # Загружаем дизайн
        self.make_flag.clicked.connect(self.run)

    def run(self):
        color1 = self.color_group_1.checkedButton().text()
        color2 = self.color_group_2.checkedButton().text()
        color3 = self.color_group_3.checkedButton().text()
        self.result.setText(f'Цвета: {color1}, {color2} и {color3}')


if __name__ == '__main__':
    if hasattr(QtCore.Qt, 'AA_EnableHighDpiScaling'):
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    if hasattr(QtCore.Qt, 'AA_UseHighDpiPixmaps'):
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)

    sys._excepthook = sys.excepthook


    def exception_hook(exctype, value, traceback):
        print(exctype, value, traceback)
        sys._excepthook(exctype, value, traceback)
        sys.exit(1)


    sys.excepthook = exception_hook

    app = QApplication(sys.argv)
    w = FlagMaker()
    w.show()
    sys.exit(app.exec())
