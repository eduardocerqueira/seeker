#date: 2023-02-03T17:05:52Z
#url: https://api.github.com/gists/ec4f06e2809a6a9f0a34a8048d7c171c
#owner: https://api.github.com/users/mdmitry1

#!/usr/bin/python3.11
'''
https://www.pythonguis.com/tutorials/qtableview-modelviews-numpy-pandas/
'''
from os.path import realpath, basename
from sys import argv, exit
from rich import print as rprint
from PyQt5.QtWidgets import QMainWindow, QApplication, QTableView, QHeaderView
from PyQt5.QtCore import Qt, QAbstractTableModel
from argparse import ArgumentParser
from pandas import read_csv

class TableModel(QAbstractTableModel):
    def __init__(self, data):
        super(TableModel, self).__init__()
        self._data = data
    def data(self, index, role): 
        if role == Qt.DisplayRole: return str(self._data.iloc[index.row(), index.column()])
    def rowCount(self, index): return self._data.shape[0]
    def columnCount(self, index): return self._data.shape[1]
    def headerData(self, section, orientation, role):
        # section is the index of the column/row.
        if role == Qt.DisplayRole: return str(self._data.columns[section]) if orientation == Qt.Horizontal \
                                          else str(self._data.index[section]+1)

class MainWindow(QMainWindow):
    def __init__(self,args, script_name):
        super().__init__()
        self.table = QTableView()
        self.table.setStyleSheet('font-size: 14px;')
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents) 
        try:
            self.df = read_csv(args.file,sep=args.separator) \
                if args.header else read_csv(args.file,sep=args.separator,header=None) 
        except Exception as err:
            rprint("\n[magenta] " + script_name + ":", "[red] ERROR: [/red]", "[red] " + str(err), "\n")
            exit(1)
        self.model = TableModel(self.df)
        self.table.setModel(self.model)
        self.setCentralWidget(self.table)
        self.table.doubleClicked.connect(self.table_view_doubleClicked)
        self.resize(900,150)
    def table_view_doubleClicked(self, index):
        row = index.row()
        column = index.column()
        value=self.df.iloc[row,column]
        print(f"Row: {row+1}, Column: {column+1}, value = {value}")

def main():
    parser = ArgumentParser()
    parser.add_argument('--file', '-f', default="/dev/stdin")
    parser.add_argument('--header', '-hdr', default=None, action='store_true')
    parser.add_argument('--separator', '-s', default='\s+')
    args=parser.parse_args()
    app=QApplication(argv)
    window=MainWindow(args, basename(realpath(argv[0])))
    window.show()
    exit(app.exec_())

if __name__ == '__main__':
    main()
