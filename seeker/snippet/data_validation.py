#date: 2022-03-18T16:49:03Z
#url: https://api.github.com/gists/19a8c15e24ccd845314a0445fb84b63b
#owner: https://api.github.com/users/marksparrish

def data_contract():
    return {
    'good_column_header': ['possible_header_1', 'possible_header_2],
    'another_good_column': ['only_one_possible_header']
  
def _validate_data_contract(df):
      vdf = pd.DataFrame(data=None)
      df_columns = df.columns
      for key, possible_headers in data_contract().items():
          # keep if any header in headers is in the df.columns
          in_df = False
          for heading in possible_headers:
              # capture heading as the column we need in the validated df (vdf)
              if heading in df_columns:
                  in_df = heading

          if in_df:
              vdf[key] = df[in_df].str.strip()
          else:
              raise ValueError(f"Broken Contract!@! for {key, possible_headers, df_columns}")
      return vdf