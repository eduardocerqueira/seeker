#date: 2022-03-01T16:50:02Z
#url: https://api.github.com/gists/3fb9585278ecfd74ab397a11c5b522de
#owner: https://api.github.com/users/Atheprogrammer

import tkinter as tk
from  misskey import Misskey
import logging

""" made by technoshy t
description:mnissskeu client for all des 
"""
instancemiss = "stop.voring.me"
class MainWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        """base ui"""
        self.lblAppTitle = tk.Label(self,text="missdesktop")
        self.lblAppTitle.grid(row=1,column=2)
        self.testButton = tk.Button(self,text="test app",command=self.TestMiss)
        self.testButton.grid(row=1,column=3)
        self.connectionTest = tk.Label(self,text="not tested")
    def TestMiss(self):
        try:
            misskey = Misskey()
            misskey.notes_create("test done")
        except:
            








mainwindow = MainWindow()
mainwindow.mainloop()