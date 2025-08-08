#date: 2025-08-08T17:03:25Z
#url: https://api.github.com/gists/336622841f2667101aa8c84e6fe029e5
#owner: https://api.github.com/users/datavudeja

#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import xlsxwriter

# Set filename tag
tag = 'client'

# Set filepaths
data_folder = Path.cwd().joinpath('data').joinpath(tag)
output_folder = Path.cwd().joinpath('output')
source_files = [f for f in data_folder.glob('*.xls*')]

def make_df(files):
    df = pd.concat(
         pd.read_excel(f,
            #sep='c0n\\$ult@xe',
            header=0,
            encoding='latin-1',
            thousands=' ',
            decimal=',',
            index_col=False,
            quotechar='"',
            #names=names,
            # skipfooter=1
            ) for f in files)
    return df


# Create and fix dataframe data types
df = make_df(source_files)
int_cols = list(df.select_dtypes(include=['int64']).columns)
df[int_cols] = df[int_cols].to_string()
float_cols = list(df.select_dtypes(include=['float64']).columns)
df[float_cols] = df[float_cols].to_string
money_cols = [
    'Montant des paiements',
    'Mnt Facture',
    'Mnt avant taxe',
    'Mnt Article',
    'Mnt TPS',
    'Mnt TVQ',
    'TPS réclamé',
    'TVQ Réclamé',
    'Taxe réclamé'
]

# Export dataframe to an Excel file

def get_column_rng(df, col_name):
        """
        Get the corresponding Excel whole-column range address from the column name.
        Needs the xlsxwriter.utility package

        Attributes:
            df: `Pandas df`
                A dataframe
            col_name: `str`
                Name of the column
        Example:
            $col_rng = get_column_letter('Numéro compte')
            $print(col_rng)
            'B:B'
        
        Returns:
            Column range as str
        """
        from xlsxwriter.utility import xl_col_to_name
        cols = list(df.columns)
        idx = cols.index(col_name)
        col_letter = xlsxwriter.utility.xl_col_to_name(idx)
        return(col_letter + ":" + col_letter)


def output_to_excel(df, output_file, currency_cols=money_cols):
    # df: a pandas dataframe
    # no_match: dataframe with data that didn't match
    # descriptor: a string to be added to the sheet name

    #Create writer
    writer = pd.ExcelWriter(output_file, engine='xlsxwriter',
                        date_format='yyyy/mm/dd',
                        datetime_format='yyyy/mm/dd')
    
    workbook  = writer.book
    # Money format
    money_format = workbook.add_format({'num_format': '0.00'})
    col_list = df.columns.to_numpy().tolist()
    
    df.to_excel(writer, sheet_name = 'Data', index = False)
    worksheet = writer.sheets[sheet]
    # Apply currency format to money columns
    for col in money_cols:
        if col in df.columns:
            col_rng = get_column_rng(df, col)
            worksheet.set_column(col_rng, None, money_format)
               
    # Set workbook properties
    workbook.set_properties({
            'title': 'CHU de Québec',
            'company': 'Consultaxe',
            'comments': 'Created with Python, Pandas and XlsxWriter'})
    workbook.set_custom_property('Client', 'CHU de Québec')

    writer.save()
    return print('Excel file created.')

filename = 'spreadsheet.xlsx'
output_to_excel(df, filename)