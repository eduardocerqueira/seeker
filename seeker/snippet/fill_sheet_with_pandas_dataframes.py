#date: 2025-06-02T16:49:41Z
#url: https://api.github.com/gists/c47e1aa688643f8bf76d235107c35736
#owner: https://api.github.com/users/bolablg

def fill_sheet_with_pandas_dataframes(sheet_id, data_frames_dict, service_account_creds):
    '''
    This function will fill a Google Sheet with data.
    Args:
        sheet_id (str): The ID of the Google Sheet to fill with data.
        data_frames_dict (dict): The data to fill the Google Sheet with. It should be a dictionary where keys are sheet names and values are pandas DataFrames.
        service_account_creds (service_account.Credentials): The service account credentials to authenticate with Google Sheets API.
    '''
    # Authorize the Google Sheets API
    gw_client = gspread.authorize(credentials=service_account_creds)
    workbook = gw_client.open_by_key(sheet_id)
    for sheet_name, df in data_frames_dict.items():
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Data for sheet '{sheet_name}' must be a pandas DataFrame.")
        
        # Clear the existing content of the sheet
        try:
            worksheet = workbook.worksheet(sheet_name)
            worksheet.clear()
        except gspread.WorksheetNotFound:
            worksheet = workbook.add_worksheet(title=sheet_name, rows="100", cols="20")
        
        # Set the DataFrame to the Google Sheet
        set_with_dataframe(worksheet, df, include_index=False, include_column_header=True)
        print(f"Data for sheet '{sheet_name}' has been written to the Google Sheet.")
    
    sheet_to_delete = workbook.worksheet("Sheet1") # Delete the default sheet created by Google Sheets
    workbook.del_worksheet(sheet_to_delete)