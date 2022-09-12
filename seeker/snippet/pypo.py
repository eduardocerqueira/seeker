#date: 2022-09-12T17:05:09Z
#url: https://api.github.com/gists/d82961006acf89b2cb0eefcc4544d68b
#owner: https://api.github.com/users/jiweiliew

# pypo.py
import os
import time
import tkinter as tk
import tkinter.filedialog as tkfd
import pandas as pd
from pandas.core.base import PandasObject
from pywintypes import com_error
import win32com.client as w32

def timestr():
    ''' Returns time as string: `YYYYMMDD_HHMMSS`
    '''
    return time.strftime('%Y%m%d_%H%M%S')
  
def tk_ask(method, **kw):
    ''' Connects the filedialog methods to a root to get rid
        of the stray root window.
    '''
    root = tk.Tk()
    root.withdraw()
    root.update()
    path = getattr(tkfd, method)(**kw)
    root.destroy()
    return path
   
def askopenfilenamep(**kw):
    return tk_ask('askopenfilename', **kw)
  
def read_excelp(filename=None, **kw):
    ''' If the user did not provide a file, open the filedialog
        and prompt the user to pick a file
    '''
    if not filename:
        filename = askopenfilenamep()
        
    if kw.get('inspect', True): # <- default to `True`
        xlsFile = pd.ExcelFile(filename)
        if len(xlsFile.sheet_names)>1:
            print('More than 1 sheet found.')
            
    for sheet in xlsFile.sheet_names:
        df = pd.read_excel(filename, sheet_name=sheet)
        print('Sheet `{}`: {}'.format(sheet, df.shape))
            
    return pd.read_excel(filename, **kw)
  
def to_excelp(df, *arg, **kw):
    ''' Writes dataframe to Excel and opens it.
    '''
  
    def xlopen(path):
        '''Opens the file (.csv, .xls, .xlsx) in Excel
        '''
        
        xl = w32.Dispatch('Excel.Application')
        try:
            wb = xl.Workbooks.Open(path)
        except com_error:
            print('Checking if file exists in current working directory...', end='')
            
            if path in os.listdir():
                print('found!')
                path = '{}\{}'.format(os.getcwd(), path)
                wb = xl.Workbooks.Open(path)
                
            else:
                print('not found! Please check file path!')
                
        else:
            pass
        finally:
            xl.Visible = True
        return path, xl, wb
      
    filename, *arg = arg
    # Give it an Excel extension
    if not filename.endswith(('.xlsx','.xls','.xlsm')):
        filename += '.xlsx'
    # Rename output file to avoid filename clashes.
    # Caveat: If there are 2 files generated in the same 
    #         second, there will still be an error
    
    if os.path.isfile(filename):
        name, ext = filename.rsplit('.')
        filename = '{}_{}.{}'.format(name, timestr(), ext)
        
    # Default index=False
    index = kw.get('index', False)
    if not index:
        kw['index']=False
        
    df.to_excel(filename, *arg, **kw)
    path, xl, wb = xlopen(filename)
    return path, xl, wb
  
PandasObject.to_excelp = to_excelp