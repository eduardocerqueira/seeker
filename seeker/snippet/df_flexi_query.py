#date: 2024-02-13T16:49:09Z
#url: https://api.github.com/gists/39ed4ed21ace770f3c4ec2406227e9d4
#owner: https://api.github.com/users/ebunt

import streamlit as st 
import pandas as pd

st.header('Flexible Data Filtering UI')

data = [
    {
        'name':'name1',
        'nickname':'nickname1',
        'date':2010,
        'loc':'loc1',
        'dep':'dep1',
        'status':'status1',
        'desc':'desc1',
        'age':21
    },
    {
        'name':'name2',
        'nickname':'nickname2',
        'date':2010,
        'loc':'loc2',
        'dep':'dep2',
        'status':'status2',
        'desc':'desc2',
        'age':22
    },
    {
        'name':'name3',
        'nickname':'nickname3',
        'date':2011,
        'loc':'loc3',
        'dep':'dep3',
        'status':'status3',
        'desc':'desc3',
        'age':23
    },
    {
        'name':'name4',
        'nickname':'nickname4',
        'date':2012,
        'loc':'loc4',
        'dep':'dep4',
        'status':'status4',
        'desc':'desc4',
        'age':24
    },
]

df = pd.DataFrame.from_records(data)

st.subheader('Data table')
st.write(df)

df = df.sort_values(by='date',ascending=True)
date_sort=df.date.unique()

with st.expander('Show UI Specification', expanded=False):
    with st.echo():
        UI_SPECIFICATION = [
            {   
                'selector': {
                    'key': 'selector_name',
                    'type': st.checkbox,
                    'label': 'Name (required)',
                    'kwargs': {'value': True, 'disabled': True},
                },
                'input': {
                    'key': 'input_name',
                    'type': st.text_input,
                    'dtype': str,
                    'label': 'Name',
                    'db_col': 'name',
                    'kwargs': {},
                },
            },
            {   
                'selector': {
                    'key': 'selector_nickname',
                    'type': st.checkbox,
                    'label': 'Nickname',
                    'kwargs': {'value': False},
                },
                'input': {
                    'key': 'input_nickname',
                    'type': st.text_input,
                    'dtype': str,
                    'label': 'Nickname',
                    'db_col': 'nickname',
                    'kwargs': {},
                },
            },
            {   
                'selector': {
                    'key': 'selector_age',
                    'type': st.checkbox,
                    'label': 'Age',
                    'kwargs': {'value': False},
                },
                'input': {
                    'key': 'input_age',
                    'type': st.number_input,
                    'dtype': int,
                    'label': 'Age',
                    'db_col': 'age',
                    'kwargs': {'min_value': 0},
                },
            },
            {   
                'selector': {
                    'key': 'selector_date',
                    'type': st.checkbox,
                    'label': 'Date',
                    'kwargs': {'value': False},
                },
                'input': {
                    'key': 'input_date',
                    'type': st.selectbox,
                    'dtype': int,
                    'label': 'Select date',
                    'db_col': 'date',
                    'kwargs': {'options': date_sort, 'index': 0},
                },
            },
            {
                'selector': {
                    'key': 'selector_dep',
                    'type': st.checkbox,
                    'label': 'Dept',
                    'kwargs': {'value': False},
                },
                'input': {
                    'key': 'input_dep',
                    'type': st.multiselect,
                    'dtype': list,
                    'label': 'Select department',
                    'db_col': 'dep',
                    'kwargs': {'options': df['dep'].unique()},
                },
            },
            {
                'selector': {
                    'key': 'selector_status',
                    'type': st.checkbox,
                    'label': 'Status',
                    'kwargs': {'value': False},
                },
                'input': {
                    'key': 'input_status',
                    'type': st.multiselect,
                    'dtype': list,
                    'label': 'Select status',
                    'db_col': 'status',
                    'kwargs': {'options': df['status'].unique()},
                },
            },
            {
                'selector': {
                    'key': 'selector_location',
                    'type': st.checkbox,
                    'label': 'Location',
                    'kwargs': {'value': False},
                },
                'input': {
                    'key': 'input_location',
                    'type': st.multiselect,
                    'dtype': list,
                    'label': 'Select location',
                    'db_col': 'loc',
                    'kwargs': {'options': df['loc'].unique()},
                },
            },
        ]

def render_selectors(ui_spec):
    field_cols = st.columns([1]*len(ui_spec))
    for i, spec in enumerate(ui_spec):
        selector = spec['selector']
        with field_cols[i]:
            selector['type'](selector['label'], key=selector['key'], **selector['kwargs'])

def get_selector_values(ui_spec):
    values = {}
    for spec in ui_spec:
        selector = spec['selector']
        values[selector['key']] = {
            'label': selector['label'], 
            'value': st.session_state[selector['key']],
        }
    return values

def render_inputs(ui_spec, selector_values):
    for spec in ui_spec:
        input = spec['input']
        selector_value = selector_values[spec['selector']['key']]['value']
        if selector_value == True:
            input['type'](input['label'], key=input['key'], **input['kwargs'])

def get_input_values(ui_spec, selector_values):
    values = {}
    for spec in ui_spec:
        input = spec['input']
        selector_value = selector_values[spec['selector']['key']]['value']
        if selector_value == True:
            values[input['key']] = {
                'label': input['label'], 
                'db_col': input['db_col'], 
                'value': st.session_state[input['key']],
                'dtype': input['dtype'],
            }
    return values

st.subheader('Filter fields selection')
render_selectors(UI_SPECIFICATION)
selector_values = get_selector_values(UI_SPECIFICATION)
# st.write(selector_values)

st.subheader('Filter field inputs')
render_inputs(UI_SPECIFICATION, selector_values)
input_values = get_input_values(UI_SPECIFICATION, selector_values)
# st.write(input_values)

def build_query(input_values, logical_op, compare_op):
    query_frags = []
    for k, v in input_values.items():
        if v['dtype'] == list:
            query_frag_expanded = [f"{v['db_col']} {compare_op} '{val}'" for val in v['value']]
            query_frag = f' {logical_op} '.join(query_frag_expanded)
        elif v['dtype'] == int or v['dtype'] == float:
            query_frag = f"{v['db_col']} {compare_op} {v['dtype'](v['value'])}"
        elif v['dtype'] == str:
            query_frag = f"{v['db_col']}.str.contains('{v['dtype'](v['value'])}')"
        else:
            query_frag = f"{v['db_col']} {compare_op} '{v['dtype'](v['value'])}')"
        query_frags.append(query_frag)
    query = f' {logical_op} '.join(query_frags)
    return query

def display_results(df_results):
    st.write(df_results)

st.subheader('Query builder')

def configure_query():
    c1, c2, _ = st.columns([1,1,2])
    with c1:
        logical_op = st.selectbox('Logical operator', options=['and', 'or'], index=1)
    with c2:
        compare_op = st.selectbox('Comparator operator', options=['==', '>', '<', '<=', '>='], index=0)
    return logical_op, compare_op

logical_op, compare_op = configure_query()
query = build_query(input_values, logical_op, compare_op)

if st.checkbox('Show filter query', True):
    st.write(f'Query: `{query}`')
st.markdown('---')
if st.button("🔍 Apply filter query"):
    df_results = df.query(query, engine='python') 
    st.subheader('Filtered data results')
    display_results(df_results)

