#date: 2022-03-30T17:01:43Z
#url: https://api.github.com/gists/5cb2f68d88ce8addd55061715a544687
#owner: https://api.github.com/users/maycon2-0


import pandas as pd
from openpyxl import Workbook
import cx_Oracle
import sys
from sqlalchemy import create_engine
from PyQt6 import QtCore, QtGui, QtWidgets
import ctypes
import time
import threading
import qdarktheme

import cgitb
cgitb.enable(format = 'text')

dsn_tns = cx_Oracle.makedsn('ip-banco-oracle', 'porta', service_name='nomedoservico')
conn = cx_Oracle.connect(user=r'usuario', password='senha', dsn=dsn_tns)
c = conn.cursor()

engine = create_engine('sqlite://', echo=False)

class Ui_ConferenciadeNotas(object):
    def setupUi(self, ConferenciadeNotas):
        ConferenciadeNotas.setObjectName("ConferenciadeNotas")
        ConferenciadeNotas.resize(868, 650)
        ConferenciadeNotas.setWindowIcon(QtGui.QIcon("icone.ico"))
        self.localArquivo = QtWidgets.QTextEdit(ConferenciadeNotas)
        self.localArquivo.setGeometry(QtCore.QRect(100, 60, 590, 30))
        self.localArquivo.setObjectName("localArquivo")
        self.label = QtWidgets.QLabel(ConferenciadeNotas)
        self.label.setGeometry(QtCore.QRect(0, 0, 870, 40))
        font = QtGui.QFont()
        font.setFamily("Century Gothic")
        font.setPointSize(18)
        self.label.setFont(font)
        self.label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(ConferenciadeNotas)
        self.label_2.setGeometry(QtCore.QRect(10, 60, 90, 30))
        font = QtGui.QFont()
        font.setFamily("Century Gothic")
        font.setPointSize(16)
        font.setBold(False)
        font.setWeight(50)
        self.label_2.setFont(font)
        self.label_2.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeading|QtCore.Qt.AlignmentFlag.AlignLeft|QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.label_2.setObjectName("label_2")
        self.localizarArquivoBT = QtWidgets.QPushButton(ConferenciadeNotas)
        self.localizarArquivoBT.setGeometry(QtCore.QRect(700, 60, 160, 30))
        font = QtGui.QFont()
        font.setFamily("Century Gothic")
        font.setPointSize(12)
        self.localizarArquivoBT.setFont(font)
        self.localizarArquivoBT.setObjectName("localizarArquivoBT")
        self.localizarArquivoBT.clicked.connect(self.locArquivo)
        self.conferidoFiliais = QtWidgets.QTableWidget(ConferenciadeNotas)
        self.conferidoFiliais.setGeometry(QtCore.QRect(20, 130, 180, 440))
        font = QtGui.QFont()
        font.setFamily("Century Gothic")
        self.conferidoFiliais.setFont(font)
        self.conferidoFiliais.setRowCount(16)
        self.conferidoFiliais.setObjectName("conferidoFiliais")
        self.conferidoFiliais.setColumnCount(3)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.conferidoFiliais.setVerticalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.conferidoFiliais.setVerticalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.conferidoFiliais.setVerticalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.conferidoFiliais.setVerticalHeaderItem(3, item)
        item = QtWidgets.QTableWidgetItem()
        self.conferidoFiliais.setVerticalHeaderItem(4, item)
        item = QtWidgets.QTableWidgetItem()
        self.conferidoFiliais.setVerticalHeaderItem(5, item)
        item = QtWidgets.QTableWidgetItem()
        self.conferidoFiliais.setVerticalHeaderItem(6, item)
        item = QtWidgets.QTableWidgetItem()
        self.conferidoFiliais.setVerticalHeaderItem(7, item)
        item = QtWidgets.QTableWidgetItem()
        self.conferidoFiliais.setVerticalHeaderItem(8, item)
        item = QtWidgets.QTableWidgetItem()
        self.conferidoFiliais.setVerticalHeaderItem(9, item)
        item = QtWidgets.QTableWidgetItem()
        self.conferidoFiliais.setVerticalHeaderItem(10, item)
        item = QtWidgets.QTableWidgetItem()
        self.conferidoFiliais.setVerticalHeaderItem(11, item)
        item = QtWidgets.QTableWidgetItem()
        self.conferidoFiliais.setVerticalHeaderItem(12, item)
        item = QtWidgets.QTableWidgetItem()
        self.conferidoFiliais.setVerticalHeaderItem(13, item)
        item = QtWidgets.QTableWidgetItem()
        self.conferidoFiliais.setVerticalHeaderItem(14, item)
        item = QtWidgets.QTableWidgetItem()
        self.conferidoFiliais.setVerticalHeaderItem(15, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.conferidoFiliais.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.conferidoFiliais.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.conferidoFiliais.setHorizontalHeaderItem(2, item)

        item = QtWidgets.QTableWidgetItem()
        self.conferidoFiliais.setItem(0, 0, item)
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setKerning(True)
        item.setFont(font)
        item.setCheckState(QtCore.Qt.CheckState.Unchecked)
        self.conferidoFiliais.setItem(0, 1, item)
        item = QtWidgets.QTableWidgetItem()
        item.setCheckState(QtCore.Qt.CheckState.Unchecked)
        self.conferidoFiliais.setItem(0, 2, item)
        item = QtWidgets.QTableWidgetItem()
        self.conferidoFiliais.setItem(1, 0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        item.setCheckState(QtCore.Qt.CheckState.Unchecked)
        self.conferidoFiliais.setItem(1, 1, item)
        item = QtWidgets.QTableWidgetItem()
        item.setCheckState(QtCore.Qt.CheckState.Unchecked)
        self.conferidoFiliais.setItem(1, 2, item)
        item = QtWidgets.QTableWidgetItem()
        self.conferidoFiliais.setItem(2, 0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setCheckState(QtCore.Qt.CheckState.Unchecked)
        self.conferidoFiliais.setItem(2, 1, item)
        item = QtWidgets.QTableWidgetItem()
        item.setCheckState(QtCore.Qt.CheckState.Unchecked)
        self.conferidoFiliais.setItem(2, 2, item)
        item = QtWidgets.QTableWidgetItem()
        self.conferidoFiliais.setItem(3, 0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setCheckState(QtCore.Qt.CheckState.Unchecked)
        self.conferidoFiliais.setItem(3, 1, item)
        item = QtWidgets.QTableWidgetItem()
        item.setCheckState(QtCore.Qt.CheckState.Unchecked)
        self.conferidoFiliais.setItem(3, 2, item)
        item = QtWidgets.QTableWidgetItem()
        self.conferidoFiliais.setItem(4, 0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setCheckState(QtCore.Qt.CheckState.Unchecked)
        self.conferidoFiliais.setItem(4, 1, item)
        item = QtWidgets.QTableWidgetItem()
        item.setCheckState(QtCore.Qt.CheckState.Unchecked)
        self.conferidoFiliais.setItem(4, 2, item)
        item = QtWidgets.QTableWidgetItem()
        self.conferidoFiliais.setItem(5, 0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setCheckState(QtCore.Qt.CheckState.Unchecked)
        self.conferidoFiliais.setItem(5, 1, item)
        item = QtWidgets.QTableWidgetItem()
        item.setCheckState(QtCore.Qt.CheckState.Unchecked)
        self.conferidoFiliais.setItem(5, 2, item)
        item = QtWidgets.QTableWidgetItem()
        self.conferidoFiliais.setItem(6, 0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setCheckState(QtCore.Qt.CheckState.Unchecked)
        self.conferidoFiliais.setItem(6, 1, item)
        item = QtWidgets.QTableWidgetItem()
        item.setCheckState(QtCore.Qt.CheckState.Unchecked)
        self.conferidoFiliais.setItem(6, 2, item)
        item = QtWidgets.QTableWidgetItem()
        self.conferidoFiliais.setItem(7, 0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setCheckState(QtCore.Qt.CheckState.Unchecked)
        self.conferidoFiliais.setItem(7, 1, item)
        item = QtWidgets.QTableWidgetItem()
        item.setCheckState(QtCore.Qt.CheckState.Unchecked)
        self.conferidoFiliais.setItem(7, 2, item)
        item = QtWidgets.QTableWidgetItem()
        self.conferidoFiliais.setItem(8, 0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setCheckState(QtCore.Qt.CheckState.Unchecked)
        self.conferidoFiliais.setItem(8, 1, item)
        item = QtWidgets.QTableWidgetItem()
        item.setCheckState(QtCore.Qt.CheckState.Unchecked)
        self.conferidoFiliais.setItem(8, 2, item)
        item = QtWidgets.QTableWidgetItem()
        self.conferidoFiliais.setItem(9, 0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setCheckState(QtCore.Qt.CheckState.Unchecked)
        self.conferidoFiliais.setItem(9, 1, item)
        item = QtWidgets.QTableWidgetItem()
        item.setCheckState(QtCore.Qt.CheckState.Unchecked)
        self.conferidoFiliais.setItem(9, 2, item)
        item = QtWidgets.QTableWidgetItem()
        self.conferidoFiliais.setItem(10, 0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setCheckState(QtCore.Qt.CheckState.Unchecked)
        self.conferidoFiliais.setItem(10, 1, item)
        item = QtWidgets.QTableWidgetItem()
        item.setCheckState(QtCore.Qt.CheckState.Unchecked)
        self.conferidoFiliais.setItem(10, 2, item)
        item = QtWidgets.QTableWidgetItem()
        self.conferidoFiliais.setItem(11, 0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setCheckState(QtCore.Qt.CheckState.Unchecked)
        self.conferidoFiliais.setItem(11, 1, item)
        item = QtWidgets.QTableWidgetItem()
        item.setCheckState(QtCore.Qt.CheckState.Unchecked)
        self.conferidoFiliais.setItem(11, 2, item)
        item = QtWidgets.QTableWidgetItem()
        self.conferidoFiliais.setItem(12, 0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setCheckState(QtCore.Qt.CheckState.Unchecked)
        self.conferidoFiliais.setItem(12, 1, item)
        item = QtWidgets.QTableWidgetItem()
        item.setCheckState(QtCore.Qt.CheckState.Unchecked)
        self.conferidoFiliais.setItem(12, 2, item)
        item = QtWidgets.QTableWidgetItem()
        self.conferidoFiliais.setItem(13, 0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setCheckState(QtCore.Qt.CheckState.Unchecked)
        self.conferidoFiliais.setItem(13, 1, item)
        item = QtWidgets.QTableWidgetItem()
        item.setCheckState(QtCore.Qt.CheckState.Unchecked)
        self.conferidoFiliais.setItem(13, 2, item)
        item = QtWidgets.QTableWidgetItem()
        self.conferidoFiliais.setItem(14, 0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setCheckState(QtCore.Qt.CheckState.Unchecked)
        self.conferidoFiliais.setItem(14, 1, item)
        item = QtWidgets.QTableWidgetItem()
        item.setCheckState(QtCore.Qt.CheckState.Unchecked)
        self.conferidoFiliais.setItem(14, 2, item)
        item = QtWidgets.QTableWidgetItem()
        self.conferidoFiliais.setItem(15, 0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setCheckState(QtCore.Qt.CheckState.Unchecked)
        self.conferidoFiliais.setItem(15, 1, item)
        item = QtWidgets.QTableWidgetItem()
        item.setCheckState(QtCore.Qt.CheckState.Unchecked)
        self.conferidoFiliais.setItem(15, 2, item)
        self.conferidoFiliais.horizontalHeader().setDefaultSectionSize(50)
        self.conferidoFiliais.horizontalHeader().setMinimumSectionSize(50)
        self.conferidoFiliais.verticalHeader().setDefaultSectionSize(23)
        self.conferidoFiliais.verticalHeader().setMinimumSectionSize(23)
        self.nfsComErro = QtWidgets.QTableWidget(ConferenciadeNotas)
        self.nfsComErro.setGeometry(QtCore.QRect(200, 130, 651, 440))
        font = QtGui.QFont()
        font.setFamily("Century Gothic")
        self.nfsComErro.setFont(font)
        #self.nfsComErro.setRowCount(100)
        self.nfsComErro.setObjectName("nfsComErro")
        self.nfsComErro.setColumnCount(6)
        item = QtWidgets.QTableWidgetItem()
        self.nfsComErro.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.nfsComErro.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.nfsComErro.setHorizontalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.nfsComErro.setHorizontalHeaderItem(3, item)
        item = QtWidgets.QTableWidgetItem()
        self.nfsComErro.setHorizontalHeaderItem(4, item)
        item = QtWidgets.QTableWidgetItem()
        self.nfsComErro.setHorizontalHeaderItem(5, item)
        self.nfsComErro.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.nfsComErro.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectItems)
        self.label_3 = QtWidgets.QLabel(ConferenciadeNotas)
        self.label_3.setGeometry(QtCore.QRect(0, 100, 870, 20))
        font = QtGui.QFont()
        font.setFamily("Century Gothic")
        font.setPointSize(16)
        self.label_3.setFont(font)
        self.label_3.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label_3.setObjectName("label_3")

        self.exportResult = QtWidgets.QPushButton(ConferenciadeNotas)
        self.exportResult.setGeometry(QtCore.QRect(703, 600, 150, 30))
        font = QtGui.QFont()
        font.setFamily("Century Gothic")
        font.setPointSize(12)
        self.exportResult.setFont(font)
        self.exportResult.setObjectName("exportResult")
        self.exportResult.setText('Exportar')
        self.exportResult.clicked.connect(self.exportExcel)

        self.retranslateUi(ConferenciadeNotas)
        QtCore.QMetaObject.connectSlotsByName(ConferenciadeNotas)
        self.rows = 0
        self.conferidoFiliais.horizontalHeader().setStretchLastSection(True)
        self.nfsComErro.horizontalHeader().setStretchLastSection(True)
        self.conferidoFiliais.horizontalHeader().setStyleSheet(""" QHeaderView::section {padding-left: 2;
                                                                                         padding-right: -10;
                                                                                                            }""")
        self.nfsComErro.horizontalHeader().setStyleSheet(""" QHeaderView::section {padding-left: 2;
                                                                                         padding-right: -10;
                                                                                                            }""")

    def retranslateUi(self, ConferenciadeNotas):
        _translate = QtCore.QCoreApplication.translate
        ConferenciadeNotas.setWindowTitle(_translate("ConferenciadeNotas", "Conferência de Notas CIGAMxSEFAZ"))
        self.label.setText(_translate("ConferenciadeNotas", "Conferência de Notas CIGAM x SEFAZ"))
        self.label_2.setText(_translate("ConferenciadeNotas", "Arquivo:"))
        self.localizarArquivoBT.setText(_translate("ConferenciadeNotas", "Localizar Arquivo"))
        item = self.conferidoFiliais.verticalHeaderItem(0)
        item.setText(_translate("ConferenciadeNotas", " "))
        item = self.conferidoFiliais.verticalHeaderItem(1)
        item.setText(_translate("ConferenciadeNotas", " "))
        item = self.conferidoFiliais.verticalHeaderItem(2)
        item.setText(_translate("ConferenciadeNotas", " "))
        item = self.conferidoFiliais.verticalHeaderItem(3)
        item.setText(_translate("ConferenciadeNotas", " "))
        item = self.conferidoFiliais.verticalHeaderItem(4)
        item.setText(_translate("ConferenciadeNotas", " "))
        item = self.conferidoFiliais.verticalHeaderItem(5)
        item.setText(_translate("ConferenciadeNotas", " "))
        item = self.conferidoFiliais.verticalHeaderItem(6)
        item.setText(_translate("ConferenciadeNotas", " "))
        item = self.conferidoFiliais.verticalHeaderItem(7)
        item.setText(_translate("ConferenciadeNotas", " "))
        item = self.conferidoFiliais.verticalHeaderItem(8)
        item.setText(_translate("ConferenciadeNotas", " "))
        item = self.conferidoFiliais.verticalHeaderItem(9)
        item.setText(_translate("ConferenciadeNotas", " "))
        item = self.conferidoFiliais.verticalHeaderItem(10)
        item.setText(_translate("ConferenciadeNotas", " "))
        item = self.conferidoFiliais.verticalHeaderItem(11)
        item.setText(_translate("ConferenciadeNotas", " "))
        item = self.conferidoFiliais.verticalHeaderItem(12)
        item.setText(_translate("ConferenciadeNotas", " "))
        item = self.conferidoFiliais.verticalHeaderItem(13)
        item.setText(_translate("ConferenciadeNotas", " "))
        item = self.conferidoFiliais.verticalHeaderItem(14)
        item.setText(_translate("ConferenciadeNotas", " "))
        item = self.conferidoFiliais.verticalHeaderItem(15)
        item.setText(_translate("ConferenciadeNotas", " "))
        item = self.conferidoFiliais.horizontalHeaderItem(0)
        item.setText(_translate("ConferenciadeNotas", "UN"))
        item = self.conferidoFiliais.horizontalHeaderItem(1)
        item.setText(_translate("ConferenciadeNotas", "NFE"))
        item = self.conferidoFiliais.horizontalHeaderItem(2)
        item.setText(_translate("ConferenciadeNotas", "NFCE"))
        __sortingEnabled = self.conferidoFiliais.isSortingEnabled()
        self.conferidoFiliais.setSortingEnabled(False)
        item = self.conferidoFiliais.item(0, 0)
        item.setText(_translate("ConferenciadeNotas", "001"))
        item = self.conferidoFiliais.item(1, 0)
        item.setText(_translate("ConferenciadeNotas", "002"))
        item = self.conferidoFiliais.item(2, 0)
        item.setText(_translate("ConferenciadeNotas", "003"))
        item = self.conferidoFiliais.item(3, 0)
        item.setText(_translate("ConferenciadeNotas", "004"))
        item = self.conferidoFiliais.item(4, 0)
        item.setText(_translate("ConferenciadeNotas", "005"))
        item = self.conferidoFiliais.item(5, 0)
        item.setText(_translate("ConferenciadeNotas", "006"))
        item = self.conferidoFiliais.item(6, 0)
        item.setText(_translate("ConferenciadeNotas", "007"))
        item = self.conferidoFiliais.item(7, 0)
        item.setText(_translate("ConferenciadeNotas", "008"))
        item = self.conferidoFiliais.item(8, 0)
        item.setText(_translate("ConferenciadeNotas", "009"))
        item = self.conferidoFiliais.item(9, 0)
        item.setText(_translate("ConferenciadeNotas", "010"))
        item = self.conferidoFiliais.item(10, 0)
        item.setText(_translate("ConferenciadeNotas", "011"))
        item = self.conferidoFiliais.item(11, 0)
        item.setText(_translate("ConferenciadeNotas", "013"))
        item = self.conferidoFiliais.item(12, 0)
        item.setText(_translate("ConferenciadeNotas", "014"))
        item = self.conferidoFiliais.item(13, 0)
        item.setText(_translate("ConferenciadeNotas", "016"))
        item = self.conferidoFiliais.item(14, 0)
        item.setText(_translate("ConferenciadeNotas", "100"))
        item = self.conferidoFiliais.item(15, 0)
        item.setText(_translate("ConferenciadeNotas", "200"))
        self.conferidoFiliais.setSortingEnabled(__sortingEnabled)
        item = self.nfsComErro.horizontalHeaderItem(0)
        item.setText(_translate("ConferenciadeNotas", "UN"))
        item = self.nfsComErro.horizontalHeaderItem(1)
        item.setText(_translate("ConferenciadeNotas", "SERIE"))
        item = self.nfsComErro.horizontalHeaderItem(2)
        item.setText(_translate("ConferenciadeNotas", "NOTA"))
        item = self.nfsComErro.horizontalHeaderItem(3)
        item.setText(_translate("ConferenciadeNotas", "DATA"))
        item = self.nfsComErro.horizontalHeaderItem(4)
        item.setText(_translate("ConferenciadeNotas", "SITUACAO"))
        item = self.nfsComErro.horizontalHeaderItem(5)
        item.setText(_translate("ConferenciadeNotas", "TEM"))
        self.label_3.setText(_translate("ConferenciadeNotas", "Unidade: Série: Data: até "))

    def locArquivo(self):
        arquivoLocal = QtWidgets.QFileDialog.getOpenFileNames(filter='*.xls')[0]

        if (arquivoLocal == []):
            def Mbox(title, text, style):
                return ctypes.windll.user32.MessageBoxW(0, text, title, style)

            Mbox('Erro arquivo', 'Arquivo não localizado ou invalido!', 0)

        for files in arquivoLocal:
            self.localArquivo.setText(' ')
            self.localArquivo.setText(files)

            self.file = files

            df = pd.read_excel(self.file, skiprows=lambda x: x not in list(range(6, 9999)))

            sqlSerie = " SELECT DISTINCT(A.SERIE) FROM (select CASE WHEN [SÉRIE] = '3' THEN 'NFE' WHEN [SÉRIE] = '7' THEN 'NFCE' WHEN [SÉRIE] = '8' THEN '2NFCE' ELSE 'NFCE' END AS SERIE \
                        FROM NFSEFAZ) A "

            try:
                df.to_sql('NFSEFAZ', engine, if_exists='replace', index=False)
            except:
                pass
                def Mbox(title, text, style):
                    return ctypes.windll.user32.MessageBoxW(0, text, title, style)

                Mbox('Erro arquivo', 'Arquivo '+ self.file + ' invalido, favor verificar!', 0)

            try:
                serieDf = engine.execute(sqlSerie)
            except:
                pass
                def Mbox(title, text, style):
                    return ctypes.windll.user32.MessageBoxW(0, text, title, style)

                Mbox('Erro arquivo', 'Arquivo '+ self.file + ' invalido, favor verificar!', 0)

            serieFim = pd.DataFrame(serieDf, columns=['SERIE'])

            self.serieTxt = serieFim.iloc[0]['SERIE']

            try:
                self.serieTxt2 = serieFim.iloc[1]['SERIE']
            except:
                pass
                self.serieTxt2 = serieFim.iloc[0]['SERIE']

            if(self.serieTxt in ['NFCE','2NFCE']):
                file = self.file

                dff = pd.read_excel(file, skiprows=lambda x: x not in list(range(0, 6)))

                dff.to_sql('NFCESEFAZ', engine, if_exists='replace', index=False)

                ie_un = engine.execute('SELECT REPLACE(SUBSTR("SECRETARIA DE ESTADO DE FAZENDA",21,10),"-","") FROM NFCESEFAZ WHERE "SECRETARIA DE ESTADO DE FAZENDA" LIKE "%INSCRIÇÃO ESTADUAL%"')

                ie_un = ie_un.first()[0]

                df = pd.read_excel(file, skiprows=lambda x: x not in list(range(6, 9999)))

                sqlsefaz = (" select CASE WHEN {} = 130241750 THEN '001' \
                            WHEN {} = 131817086 THEN '002'\
                            WHEN {} = 131838245 THEN '003'\
                            WHEN {} = 131875523 THEN '004'\
                            WHEN {} = 131980203 THEN '005'\
                            WHEN {} = 132009412 THEN '006'\
                            WHEN {} = 132894939 THEN '007'\
                            WHEN {} = 132702371 THEN '008'\
                            WHEN {} = 133644065 THEN '009'\
                            WHEN {} = 131537326 THEN '010'\
                            WHEN {} = 133446565 THEN '011'\
                            WHEN {} = 132124726 THEN '013'\
                            WHEN {} = 133779416 THEN '014'\
                            WHEN {} = 133830900 THEN '016'\
                            WHEN {} = 133762033 THEN '100'\
                            WHEN {} = 131847031 THEN '200' ELSE {} END AS UN,\
                            CASE WHEN [SÉRIE] = '3' THEN 'NFE' WHEN [SÉRIE] = '7' THEN 'NFCE' WHEN [SÉRIE] = '8' THEN '2NFCE' ELSE 'NFCE' END AS SERIE,\
                            [NUMERO NOTA FISCAL] as NF, SUBSTR([DATA EMISSÃO],0,11) as DT_NF, \
                            CASE WHEN upper([SITUAÇÃO]) = 'CANCELADA FORA DO PRAZO' THEN 'CANCELADA' \
                                WHEN upper([SITUAÇÃO]) = 'AUTORIZADA FORA PRAZO' THEN 'AUTORIZADA' ELSE upper([SITUAÇÃO]) END AS SITUACAO\
                            FROM NFSEFAZ ").format(ie_un, ie_un, ie_un, ie_un, ie_un, ie_un, ie_un, ie_un, ie_un, ie_un,
                                                   ie_un, ie_un, ie_un, ie_un, ie_un, ie_un, ie_un)

                df.to_sql('NFSEFAZ', engine, if_exists='replace', index=False)

                results = engine.execute(sqlsefaz)

                final = pd.DataFrame(results, columns=['UN', 'SERIE', 'NF', 'DT_NF', 'SITUACAO'])

                final.to_sql('NOTASSEFAZ', engine, if_exists='replace', index=False)

                dt_inicio = engine.execute('SELECT MIN(SUBSTR([DATA EMISSÃO],0,11)) FROM NFSEFAZ')

                dt_fim = engine.execute('SELECT MAX(SUBSTR([DATA EMISSÃO],0,11)) FROM NFSEFAZ')

                un_neg = engine.execute('SELECT distinct(UN) FROM NOTASSEFAZ')

                serie_nf = engine.execute('SELECT distinct(SERIE) FROM NOTASSEFAZ')

                dt_inicio = dt_inicio.first()[0]

                dt_fim = dt_fim.first()[0]

                un_neg = un_neg.first()[0]

                #serie_nf = [dict(row) for row in serie_nf]

                list_serie = []

                for row in serie_nf:
                    list_serie.append(row[0])

                list_serie = str(list_serie)[1:-1]

                self.label_3.setText("Unidade: " + un_neg + " Série: " + list_serie.replace("'",'').replace(",",' e') + " Data: " + dt_inicio + " até " + dt_fim)

                #self.dtLabel["text"] = " Unidade: "+ un_neg + " Série: " + self.serieTxt + " Data: "+ dt_inicio+ " até " + dt_fim

                sql = ("""SELECT F.CD_UNIDADE_DE_N,\
                           F.SERIE,F.NF,TO_CHAR(F.DT_EMISSAO, 'DD/MM/YYYY') AS DT,\
                           CASE WHEN F.ESPECIE_NOTA = 'S' THEN 'AUTORIZADA' \
                                WHEN F.ESPECIE_NOTA = 'N' THEN 'CANCELADA' \
                                WHEN F.ESPECIE_NOTA = 'E' THEN 'AUTORIZADA' \
                                END AS STATUS \
                           FROM FANFISCA F \
                           WHERE F.SERIE in ({}) \
                           AND F.CD_UNIDADE_DE_N = '{}' \
                           AND F.DT_EMISSAO BETWEEN '{}' AND '{}' \
                           """).format(list_serie, un_neg, dt_inicio, dt_fim)

                nfbanco = pd.read_sql(sql, conn)

                nfbanco.to_sql('NFCIGAM', engine, if_exists='replace', index=False)

                comparaNfSefaz = engine.execute(" SELECT S.*,'SEFAZ' AS TEM FROM NOTASSEFAZ S LEFT JOIN NFCIGAM C ON (S.UN = C.CD_UNIDADE_DE_N AND S.SERIE = C.SERIE AND S.NF = C.NF) WHERE C.NF IS NULL")

                resultComparaNfSefaz = pd.DataFrame(comparaNfSefaz, columns=['UN', 'SERIE', 'NOTA', 'DATA', 'SITUACAO', 'TEM'])

                comparaNfCigam = engine.execute(" SELECT C.*,'CIGAM' AS TEM FROM NFCIGAM C LEFT JOIN NOTASSEFAZ S ON ( C.CD_UNIDADE_DE_N = S.UN AND C.SERIE = S.SERIE AND C.NF = S.NF) WHERE S.NF IS NULL")

                resultComparaNfCigam = pd.DataFrame(comparaNfCigam, columns=['UN', 'SERIE', 'NOTA', 'DATA', 'SITUACAO', 'TEM'])

                comparaNfCigamXSefaz = engine.execute( " SELECT C.CD_UNIDADE_DE_N,C.SERIE,C.NF,C.DT,C.STATUS || ' x ' || S.SITUACAO,'CIGAM e SEFAZ' as TEM FROM NFCIGAM C INNER JOIN NOTASSEFAZ S ON ( C.CD_UNIDADE_DE_N = S.UN AND C.SERIE = S.SERIE AND C.NF = S.NF) WHERE S.SITUACAO <> C.STATUS")

                resultComparaNfCigamXSefaz = pd.DataFrame(comparaNfCigamXSefaz, columns=['UN', 'SERIE', 'NOTA', 'DATA', 'SITUACAO','TEM'])

                for index, row in resultComparaNfSefaz.iterrows():
                    #print(row[0])
                    self.nfsComErro.setRowCount(self.rows+1)
                    self.nfsComErro.setItem(self.rows, 0, QtWidgets.QTableWidgetItem(str(row["UN"])))
                    self.nfsComErro.setItem(self.rows, 1, QtWidgets.QTableWidgetItem(str(row["SERIE"])))
                    self.nfsComErro.setItem(self.rows, 2, QtWidgets.QTableWidgetItem(str(row["NOTA"])))
                    self.nfsComErro.setItem(self.rows, 3, QtWidgets.QTableWidgetItem(str(row["DATA"])))
                    self.nfsComErro.setItem(self.rows, 4, QtWidgets.QTableWidgetItem(str(row["SITUACAO"])))
                    self.nfsComErro.setItem(self.rows, 5, QtWidgets.QTableWidgetItem(str(row["TEM"])))
                    self.rows=self.rows+1

                for index, row in resultComparaNfCigam.iterrows():
                    #print(row[0])
                    self.nfsComErro.setRowCount(self.rows+1)
                    self.nfsComErro.setItem(self.rows, 0, QtWidgets.QTableWidgetItem(str(row["UN"])))
                    self.nfsComErro.setItem(self.rows, 1, QtWidgets.QTableWidgetItem(str(row["SERIE"])))
                    self.nfsComErro.setItem(self.rows, 2, QtWidgets.QTableWidgetItem(str(row["NOTA"])))
                    self.nfsComErro.setItem(self.rows, 3, QtWidgets.QTableWidgetItem(str(row["DATA"])))
                    self.nfsComErro.setItem(self.rows, 4, QtWidgets.QTableWidgetItem(str(row["SITUACAO"])))
                    self.nfsComErro.setItem(self.rows, 5, QtWidgets.QTableWidgetItem(str(row["TEM"])))
                    self.rows=self.rows+1

                for index, row in resultComparaNfCigamXSefaz.iterrows():
                    #print(row[0])
                    self.nfsComErro.setRowCount(self.rows+1)
                    self.nfsComErro.setItem(self.rows, 0, QtWidgets.QTableWidgetItem(str(row["UN"])))
                    self.nfsComErro.setItem(self.rows, 1, QtWidgets.QTableWidgetItem(str(row["SERIE"])))
                    self.nfsComErro.setItem(self.rows, 2, QtWidgets.QTableWidgetItem(str(row["NOTA"])))
                    self.nfsComErro.setItem(self.rows, 3, QtWidgets.QTableWidgetItem(str(row["DATA"])))
                    self.nfsComErro.setItem(self.rows, 4, QtWidgets.QTableWidgetItem(str(row["SITUACAO"])))
                    self.nfsComErro.setItem(self.rows, 5, QtWidgets.QTableWidgetItem(str(row["TEM"])))
                    self.rows=self.rows+1

                item = QtWidgets.QTableWidgetItem()
                item.setCheckState(QtCore.Qt.CheckState.Checked)

                if(un_neg == '001'):
                    self.conferidoFiliais.setItem(0, 2, item)
                if(un_neg == '002'):
                    self.conferidoFiliais.setItem(1, 2, item)
                if(un_neg == '003'):
                    self.conferidoFiliais.setItem(2, 2, item)
                if(un_neg == '004'):
                    self.conferidoFiliais.setItem(3, 2, item)
                if(un_neg == '005'):
                    self.conferidoFiliais.setItem(4, 2, item)
                if(un_neg == '006'):
                    self.conferidoFiliais.setItem(5, 2, item)
                if(un_neg == '007'):
                    self.conferidoFiliais.setItem(6, 2, item)
                if(un_neg == '008'):
                    self.conferidoFiliais.setItem(7, 2, item)
                if(un_neg == '009'):
                    self.conferidoFiliais.setItem(8, 2, item)
                if(un_neg == '010'):
                    self.conferidoFiliais.setItem(9, 2, item)
                if(un_neg == '011'):
                    self.conferidoFiliais.setItem(10, 2, item)
                if(un_neg == '013'):
                    self.conferidoFiliais.setItem(11, 2, item)
                if(un_neg == '014'):
                    self.conferidoFiliais.setItem(12, 2, item)
                if(un_neg == '016'):
                    self.conferidoFiliais.setItem(13, 2, item)
                if(un_neg == '100'):
                    self.conferidoFiliais.setItem(14, 2, item)
                if(un_neg == '200'):
                    self.conferidoFiliais.setItem(15, 2, item)

                def worker(title, close_until_seconds):
                    time.sleep(close_until_seconds)
                    wd = ctypes.windll.user32.FindWindowW(0, title)
                    ctypes.windll.user32.SendMessageW(wd, 0x0010, 0, 0)
                    return

                def AutoCloseMessageBoxW(text, title,  close_until_seconds):
                    t = threading.Thread(target=worker, args=(title, close_until_seconds))
                    t.start()
                    ctypes.windll.user32.MessageBoxW(0, text, title, 0)

                AutoCloseMessageBoxW('Conferido NFCe UN:'+un_neg, 'NFCe Conferida', 0.5)


                    #print(resultComparaNfSefaz, "\n", "\n", resultComparaNfCigam, "\n", "\n", resultComparaNfCigamXSefaz)


            if (self.serieTxt == 'NFE'):
                file = self.file

                df = pd.read_excel(file, skiprows=lambda x: x not in list(range(6, 9999)))

                sqlsefaz = " select CASE WHEN [INSCRIÇÃO ESTADUAL] = '130241750' THEN '001' \
                            WHEN [INSCRIÇÃO ESTADUAL] = '131817086' THEN '002'\
                            WHEN [INSCRIÇÃO ESTADUAL] = '131838245' THEN '003'\
                            WHEN [INSCRIÇÃO ESTADUAL] = '131875523' THEN '004'\
                            WHEN [INSCRIÇÃO ESTADUAL] = '131980203' THEN '005'\
                            WHEN [INSCRIÇÃO ESTADUAL] = '132009412' THEN '006'\
                            WHEN [INSCRIÇÃO ESTADUAL] = '132894939' THEN '007'\
                            WHEN [INSCRIÇÃO ESTADUAL] = '132702371' THEN '008'\
                            WHEN [INSCRIÇÃO ESTADUAL] = '133644065' THEN '009'\
                            WHEN [INSCRIÇÃO ESTADUAL] = '131537326' THEN '010'\
                            WHEN [INSCRIÇÃO ESTADUAL] = '133446565' THEN '011'\
                            WHEN [INSCRIÇÃO ESTADUAL] = '132124726' THEN '013'\
                            WHEN [INSCRIÇÃO ESTADUAL] = '133779416' THEN '014'\
                            WHEN [INSCRIÇÃO ESTADUAL] = '133830900' THEN '016'\
                            WHEN [INSCRIÇÃO ESTADUAL] = '133762033' THEN '100'\
                            WHEN [INSCRIÇÃO ESTADUAL] = '131847031' THEN '200' ELSE [INSCRIÇÃO ESTADUAL] END AS UN,\
                            CASE WHEN [SÉRIE] = '3' THEN 'NFE' WHEN [SÉRIE] = '7' THEN 'NFCE' WHEN [SÉRIE] = '8' THEN '2NFCE' ELSE 'NFCE' END AS SERIE,\
                            [NUMERO NOTA FISCAL] as NF, SUBSTR([DATA EMISSÃO],0,11) as DT_NF, \
                            CASE WHEN [SITUAÇÃO] = 'CANCELADA FORA DO PRAZO' THEN 'CANCELADA' \
                                 WHEN [SITUAÇÃO] = 'AUTORIZADA FORA DO PRAZO' THEN 'AUTORIZADA' \
                                    ELSE [SITUAÇÃO]\
                            END AS SITUACAO\
                            FROM NFSEFAZ "

                df.to_sql('NFSEFAZ', engine, if_exists='replace', index=False)

                results = engine.execute(sqlsefaz)

                final = pd.DataFrame(results, columns=['UN', 'SERIE', 'NF', 'DT_NF', 'SITUACAO'])

                final.to_sql('NOTASSEFAZ', engine, if_exists='replace', index=False)

                dt_inicio = engine.execute('SELECT MIN(SUBSTR([DATA EMISSÃO],0,11)) FROM NFSEFAZ')

                dt_fim = engine.execute('SELECT MAX(SUBSTR([DATA EMISSÃO],0,11)) FROM NFSEFAZ')

                un_neg = engine.execute('SELECT distinct(UN) FROM NOTASSEFAZ')

                serie_nf = engine.execute('SELECT distinct(SERIE) FROM NOTASSEFAZ')

                dt_inicio = dt_inicio.first()[0]

                dt_fim = dt_fim.first()[0]

                un_neg = un_neg.first()[0]

                serie_nf = serie_nf.first()[0]

                self.label_3.setText("Unidade: " + un_neg + " Série: " + serie_nf + " Data: " + dt_inicio + " até " + dt_fim)

                #self.dtLabel["text"] = " Unidade: " + un_neg + " Série: " + self.serieTxt + " Data: " + dt_inicio + " até " + dt_fim

                sql = ("""SELECT F.CD_UNIDADE_DE_N,\
                           F.SERIE,F.NF,TO_CHAR(F.DT_EMISSAO, 'DD/MM/YYYY') AS DT,\
                           CASE WHEN F.ESPECIE_NOTA = 'S' THEN 'AUTORIZADA' \
                                WHEN F.ESPECIE_NOTA = 'N' THEN 'CANCELADA' \
                                WHEN F.ESPECIE_NOTA = 'E' THEN 'AUTORIZADA' \
                                END AS STATUS \
                           FROM FANFISCA F \
                           WHERE F.SERIE = '{}' \
                           AND F.CD_UNIDADE_DE_N = '{}' \
                           AND F.DT_EMISSAO BETWEEN '{}' AND '{}' \
                           """).format(serie_nf, un_neg, dt_inicio, dt_fim)

                nfbanco = pd.read_sql(sql, conn)

                nfbanco.to_sql('NFCIGAM', engine, if_exists='replace', index=False)

                comparaNfSefaz = engine.execute(" SELECT S.*,'SEFAZ' AS TEM FROM NOTASSEFAZ S LEFT JOIN NFCIGAM C ON (S.UN = C.CD_UNIDADE_DE_N AND S.SERIE = C.SERIE AND S.NF = C.NF) WHERE C.NF IS NULL")

                resultComparaNfSefaz = pd.DataFrame(comparaNfSefaz, columns=['UN', 'SERIE', 'NOTA', 'DATA', 'SITUACAO', 'TEM'])

                comparaNfCigam = engine.execute(" SELECT C.*,'CIGAM' AS TEM FROM NFCIGAM C LEFT JOIN NOTASSEFAZ S ON ( C.CD_UNIDADE_DE_N = S.UN AND C.SERIE = S.SERIE AND C.NF = S.NF) WHERE S.NF IS NULL")

                resultComparaNfCigam = pd.DataFrame(comparaNfCigam, columns=['UN', 'SERIE', 'NOTA', 'DATA', 'SITUACAO', 'TEM'])

                comparaNfCigamXSefaz = engine.execute(" SELECT C.CD_UNIDADE_DE_N,C.SERIE,C.NF,C.DT,C.STATUS||' x '||S.SITUACAO,'CIGAM E SEFAZ' AS TEM FROM NFCIGAM C INNER JOIN NOTASSEFAZ S ON ( C.CD_UNIDADE_DE_N = S.UN AND C.SERIE = S.SERIE AND C.NF = S.NF) WHERE S.SITUACAO <> C.STATUS")

                resultComparaNfCigamXSefaz = pd.DataFrame(comparaNfCigamXSefaz, columns=['UN', 'SERIE', 'NOTA', 'DATA', 'SITUACAO', 'TEM'])


                for index, row in resultComparaNfSefaz.iterrows():
                    #print(row[0])
                    self.nfsComErro.setRowCount(self.rows+1)
                    self.nfsComErro.setItem(self.rows, 0, QtWidgets.QTableWidgetItem(str(row["UN"])))
                    self.nfsComErro.setItem(self.rows, 1, QtWidgets.QTableWidgetItem(str(row["SERIE"])))
                    self.nfsComErro.setItem(self.rows, 2, QtWidgets.QTableWidgetItem(str(row["NOTA"])))
                    self.nfsComErro.setItem(self.rows, 3, QtWidgets.QTableWidgetItem(str(row["DATA"])))
                    self.nfsComErro.setItem(self.rows, 4, QtWidgets.QTableWidgetItem(str(row["SITUACAO"])))
                    self.nfsComErro.setItem(self.rows, 5, QtWidgets.QTableWidgetItem(str(row["TEM"])))
                    self.rows=self.rows+1

                for index, row in resultComparaNfCigam.iterrows():
                    #print(row[0])
                    self.nfsComErro.setRowCount(self.rows+1)
                    self.nfsComErro.setItem(self.rows, 0, QtWidgets.QTableWidgetItem(str(row["UN"])))
                    self.nfsComErro.setItem(self.rows, 1, QtWidgets.QTableWidgetItem(str(row["SERIE"])))
                    self.nfsComErro.setItem(self.rows, 2, QtWidgets.QTableWidgetItem(str(row["NOTA"])))
                    self.nfsComErro.setItem(self.rows, 3, QtWidgets.QTableWidgetItem(str(row["DATA"])))
                    self.nfsComErro.setItem(self.rows, 4, QtWidgets.QTableWidgetItem(str(row["SITUACAO"])))
                    self.nfsComErro.setItem(self.rows, 5, QtWidgets.QTableWidgetItem(str(row["TEM"])))
                    self.rows=self.rows+1

                for index, row in resultComparaNfCigamXSefaz.iterrows():
                    #print(row[0])
                    self.nfsComErro.setRowCount(self.rows+1)
                    self.nfsComErro.setItem(self.rows, 0, QtWidgets.QTableWidgetItem(str(row["UN"])))
                    self.nfsComErro.setItem(self.rows, 1, QtWidgets.QTableWidgetItem(str(row["SERIE"])))
                    self.nfsComErro.setItem(self.rows, 2, QtWidgets.QTableWidgetItem(str(row["NOTA"])))
                    self.nfsComErro.setItem(self.rows, 3, QtWidgets.QTableWidgetItem(str(row["DATA"])))
                    self.nfsComErro.setItem(self.rows, 4, QtWidgets.QTableWidgetItem(str(row["SITUACAO"])))
                    self.nfsComErro.setItem(self.rows, 5, QtWidgets.QTableWidgetItem(str(row["TEM"])))
                    self.rows=self.rows+1

                item = QtWidgets.QTableWidgetItem()
                item.setCheckState(QtCore.Qt.CheckState.Checked)

                if(un_neg == '001'):
                    self.conferidoFiliais.setItem(0, 1, item)
                if(un_neg == '002'):
                    self.conferidoFiliais.setItem(1, 1, item)
                if(un_neg == '003'):
                    self.conferidoFiliais.setItem(2, 1, item)
                if(un_neg == '004'):
                    self.conferidoFiliais.setItem(3, 1, item)
                if(un_neg == '005'):
                    self.conferidoFiliais.setItem(4, 1, item)
                if(un_neg == '006'):
                    self.conferidoFiliais.setItem(5, 1, item)
                if(un_neg == '007'):
                    self.conferidoFiliais.setItem(6, 1, item)
                if(un_neg == '008'):
                    self.conferidoFiliais.setItem(7, 1, item)
                if(un_neg == '009'):
                    self.conferidoFiliais.setItem(8, 1, item)
                if(un_neg == '010'):
                    self.conferidoFiliais.setItem(9, 1, item)
                if(un_neg == '011'):
                    self.conferidoFiliais.setItem(10, 1, item)
                if(un_neg == '013'):
                    self.conferidoFiliais.setItem(11, 1, item)
                if(un_neg == '014'):
                    self.conferidoFiliais.setItem(12, 1, item)
                if(un_neg == '016'):
                    self.conferidoFiliais.setItem(13, 1, item)
                if(un_neg == '100'):
                    self.conferidoFiliais.setItem(14, 1, item)
                if(un_neg == '200'):
                    self.conferidoFiliais.setItem(15, 1, item)

                def worker(title, close_until_seconds):
                    time.sleep(close_until_seconds)
                    wd = ctypes.windll.user32.FindWindowW(0, title)
                    ctypes.windll.user32.SendMessageW(wd, 0x0010, 0, 0)
                    return

                def AutoCloseMessageBoxW(text, title, close_until_seconds):
                    t = threading.Thread(target=worker, args=(title, close_until_seconds))
                    t.start()
                    return ctypes.windll.user32.MessageBoxW(0, text, title, 0)

                AutoCloseMessageBoxW('Conferido NFe UN:'+un_neg, 'NFE Conferida', 0.5)

        ctypes.windll.user32.MessageBoxW(0, 'Conferência Finaliza', 'Conferido', 0)


    def exportExcel(self):
        columnHeaders = []

        for j in range(self.nfsComErro.model().columnCount()):
            columnHeaders.append(self.nfsComErro.horizontalHeaderItem(j).text())

        df = pd.DataFrame(columns=columnHeaders)

        for row in range(self.nfsComErro.model().rowCount()):
            for col in range(self.nfsComErro.columnCount()):
                #item =
                if item := self.nfsComErro.item(row, col).text():
                    df.at[row, columnHeaders[col]] = self.nfsComErro.item(row, col).text()

        fileExport = QtWidgets.QFileDialog.getSaveFileName(filter='*.xlsx')[0]
        #print(fileExport)
        df.to_excel(fileExport, index=False)

        def Mbox(title, text, style):
            return ctypes.windll.user32.MessageBoxW(0, text, title, style)

        Mbox('Arquivo Exportado', 'Resultado exportado para o arquivo ' + fileExport, 0)

                #print(resultComparaNfSefaz, "\n", "\n", resultComparaNfCigam, "\n", "\n", resultComparaNfCigamXSefaz)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(qdarktheme.load_stylesheet())
    font = QtGui.QFont()
    font.setFamily("Century Gothic")
    font.setPointSize(10)
    app.setFont(font)
    ConferenciadeNotas = QtWidgets.QDialog()
    ui = Ui_ConferenciadeNotas()
    ui.setupUi(ConferenciadeNotas)
    ConferenciadeNotas.show()
    sys.exit(app.exec())
