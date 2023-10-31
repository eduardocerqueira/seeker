#date: 2023-10-31T16:49:43Z
#url: https://api.github.com/gists/4f6b5debf91f4a1107c3ddb2a2352ad7
#owner: https://api.github.com/users/dharmatech

import pandas as pd
import treasury_gov_pandas
from bokeh.plotting import figure, show
from bokeh.models   import NumeralTickFormatter, HoverTool
import bokeh.models

import bokeh.palettes
import bokeh.transform

# ----------------------------------------------------------------------
df = treasury_gov_pandas.update_records(
    'revenue-collections.pkl',
    'https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v2/revenue/rcm')

def concise_columns(df=df):
    return df.drop(columns=['record_calendar_year', 'record_calendar_quarter', 'record_calendar_month', 'record_calendar_day', 'record_fiscal_year', 'record_fiscal_quarter', 'src_line_nbr'])

# print(concise_columns().tail(100).to_string())
# ----------------------------------------------------------------------

df['record_date'] = pd.to_datetime(df['record_date'])

df['net_collections_amt'] = pd.to_numeric(df['net_collections_amt'])

# temp = df[df['tax_category_desc'] == 'IRS Tax']

# print(concise_columns(temp).tail(100).to_string())

irs_tax = df[df['tax_category_desc'] == 'IRS Tax']

# irs_tax.groupby(pd.Grouper(key='record_date', freq='D'))['net_collections_amt'].sum().to_frame().index.name = 'date'

# irs_tax_by_day = irs_tax.groupby(pd.Grouper(key='record_date', freq='D'))['net_collections_amt'].sum().to_frame()

irs_tax_by_month = irs_tax.groupby(pd.Grouper(key='record_date', freq='M'))['net_collections_amt'].sum().to_frame()

irs_tax_by_year = irs_tax.groupby(pd.Grouper(key='record_date', freq='Y'))['net_collections_amt'].sum().to_frame()

# ----------------------------------------------------------------------
p = figure(
    title='U.S. Government Revenue Collections', 
    sizing_mode='stretch_both', 
    x_axis_type='datetime', 
    x_axis_label='record_date', 
    y_axis_label='net_collections_amt')

p.add_tools(HoverTool(tooltips=[
    ('record_date',    '@record_date{%F}'),
    ('net_collections_amt', '@net_collections_amt{$0.0a}'),
    ], 
    formatters={
        '@record_date':    'datetime'        
        }
    ))

p.line(x='record_date', y='net_collections_amt', color='blue', source=bokeh.models.ColumnDataSource(irs_tax_by_month), legend_label='IRX Tax : by month')
# p.line(x='record_date', y='net_collections_amt', color='blue', source=bokeh.models.ColumnDataSource(irs_tax_by_year),  legend_label='IRX Tax : by year')

p.yaxis.formatter = NumeralTickFormatter(format='$0a')

# p.legend.click_policy = 'hide'

show(p)
# ----------------------------------------------------------------------
