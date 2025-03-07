#date: 2025-03-07T16:57:54Z
#url: https://api.github.com/gists/cc08f67cdee173c54b322de44f34b782
#owner: https://api.github.com/users/goodalexander

import pandas as pd
import requests
import datetime
import csv
import io
from io import StringIO
import warnings

# Suppress the SettingWithCopyWarning
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

def get_ishares_products():
    """Get all iShares ETF products and their details"""
    api_url = 'https://www.ishares.com/us/product-screener/product-screener-v3.jsn?dcrPath=/templatedata/config/product-screener-v3/data/en/us-ishares/product-screener-ketto&siteEntryPassthrough=true'
    
    response = requests.get(api_url)
    data = pd.DataFrame(response.json()['data']['tableData']['data'])
    
    # Find the correct column indices for country, name, url, and ticker
    sample_slice = data.loc[0].copy()
    all_countries = ['New Zealand', 'Italy', 'Sweden', 'Switzerland', 'Thailand', 'Indonesia', 'China', 
                    'Philippines', 'India', 'South Korea', 'Denmark', 'Broad', 'Germany', 'Qatar', 
                    'Colombia', 'Russia', 'Hong Kong', 'Poland', 'Turkey', 'United Kingdom', 'Norway', 
                    'United Arab Emirates', 'Chile', 'Malaysia', 'Belgium', 'Brazil', 'Israel', 'Mexico', 
                    'Finland', 'Taiwan', 'Peru', 'Australia', 'Saudi Arabia', 'South Africa', 'Argentina', 
                    'Singapore', 'Austria', 'France', 'Netherlands', 'Canada', 'Ireland', 'United States', 
                    'Spain', 'Japan']
    
    country_idx = sample_slice[sample_slice.apply(lambda x: str(x) in all_countries)].index[0]
    url_idx = sample_slice[sample_slice.apply(lambda x: '/product' in str(x))].index[0]
    
    # Find ticker index
    ticker_counts = []
    for col_idx in range(0, len(data.columns)):
        ticker_count = (data[col_idx].apply(lambda x: ('IVV' in str(x)) | (('IYZ' in str(x))))).sum()
        ticker_counts.append([col_idx, ticker_count])
    ticker_idx = pd.DataFrame(ticker_counts).sort_values(1, ascending=False).iloc[0, 0]
    
    # Find name index
    name_counts = []
    for col_idx in range(0, len(data.columns)):
        name_count = (data[col_idx].apply(lambda x: ('iShares Russell 2000 Small-Cap Index Fund' in str(x)) | 
                                        (('iShares MSCI EAFE International Index Fund' in str(x))))).sum()
        name_counts.append([col_idx, name_count])
    name_idx = pd.DataFrame(name_counts).sort_values(1, ascending=False).iloc[0, 0]
    
    # Create final product dataframe
    product_df = data[[country_idx, name_idx, url_idx, ticker_idx]]
    product_df.columns = ['country', 'name', 'url', 'ticker']
    
    return product_df

def create_us_ishares_map():
    """Filter iShares ETFs to US market and equity-focused funds"""
    ishares = get_ishares_products()
    us_ishares = ishares[(ishares['country'] == "United States") | (ishares['country'] == "Broad")].copy()
    
    # Filter out bond/treasury ETFs
    us_ishares['credit'] = us_ishares['name'].apply(lambda x: ('bond' in x.lower()) | ('treasury' in x.lower()))
    us_equity_ishares = us_ishares[us_ishares['credit'] == False]
    us_equity_ishares.loc[:, 'fund_string'] = us_equity_ishares['url'].apply(lambda x: str(x).split('/products/')[1])
    
    # Create mapping from ticker to fund string for URL construction
    full_us_equity_map = us_equity_ishares.groupby('ticker').last()['fund_string'].to_dict()
    return full_us_equity_map

def get_etf_holdings(etf, date=None):
    """
    Get holdings for a specific ETF as of a specific date
    
    Parameters:
    etf (str): ETF ticker symbol (e.g., 'IWM')
    date (datetime or str, optional): Date for holdings data. If None, uses last business day.
    
    Returns:
    dict: Dictionary with 'holdings' and 'nav' dataframes
    """
    # Get the ETF URL mapping
    us_equity_map = create_us_ishares_map()
    
    if etf not in us_equity_map:
        raise ValueError(f"ETF ticker '{etf}' not found in iShares US equity ETFs")
    
    # Use provided date or default to last business day
    if date is None:
        date = pd.Timestamp.now().normalize()
        if date.dayofweek >= 5:  # If weekend, use Friday
            date = date - pd.Timedelta(days=(date.dayofweek - 4))
    elif isinstance(date, str):
        date = pd.Timestamp(date)
    
    # Format date for URL
    dt_str = date.strftime('%Y%m%d')
    
    # Construct URL for holdings data
    path = us_equity_map[etf]
    url = f'https://www.ishares.com/us/products/{path}/1467271812596.ajax?fileType=csv&fileName={etf}_membership&dataType=fund&asOfDate={dt_str}'
    
    # Get data directly without saving to file
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise exception for HTTP errors
        
        # Parse CSV directly from response content
        content_str = response.content.decode('utf-8')
        content = StringIO(content_str)
        
        # First, inspect the CSV structure to determine the format
        csv_reader = csv.reader(content)
        header_rows = []
        for i in range(20):  # Read first 20 rows to analyze structure
            try:
                header_rows.append(next(csv_reader))
            except StopIteration:
                break
        
        content.seek(0)  # Reset position
        
        # Find where the actual holdings data begins
        holdings_start_row = 0
        for i, row in enumerate(header_rows):
            # Look for rows with standard holdings headers like "Ticker" or "Name"
            if len(row) > 3 and any(header in row for header in ["Ticker", "ISIN", "Name", "Sector"]):
                holdings_start_row = i
                break
        
        # If we couldn't find a clear holdings start, use a heuristic approach
        if holdings_start_row == 0:
            # Look for the longest row, which is likely the holdings header
            holdings_start_row = max(range(len(header_rows)), key=lambda i: len(header_rows[i]))
        
        # Read the NAV summary (metadata at the top)
        content.seek(0)
        try:
            nav_summary = pd.read_csv(content, nrows=holdings_start_row-1, header=None)
        except:
            # Fallback if standard parsing fails
            nav_summary = pd.DataFrame([r for r in header_rows[:holdings_start_row-1]])
        
        # Read the holdings data with flexible parsing
        content.seek(0)
        try:
            # Try with pandas' flexible parsing
            holdings_summary = pd.read_csv(
                content, 
                skiprows=holdings_start_row,
                error_bad_lines=False,  # Skip bad lines
                warn_bad_lines=True,    # Warn about them
                low_memory=False        # Handle larger files
            )
        except:
            try:
                # Fallback method if the first approach fails
                content.seek(0)
                # Skip to the holdings section
                for _ in range(holdings_start_row):
                    next(csv_reader)
                
                # Read the remaining lines
                holdings_data = list(csv_reader)
                
                # Convert to DataFrame
                if holdings_data and len(holdings_data) > 0:
                    # Use the first row as headers
                    headers = holdings_data[0]
                    data = holdings_data[1:]
                    
                    # Create DataFrame with dynamic columns
                    holdings_summary = pd.DataFrame(data)
                    if len(holdings_summary.columns) >= len(headers):
                        holdings_summary.columns = headers + list(range(len(holdings_summary.columns) - len(headers)))
                    else:
                        # Handle case where headers are longer than data columns
                        holdings_summary.columns = headers[:len(holdings_summary.columns)]
                else:
                    # Create empty DataFrame if no data
                    holdings_summary = pd.DataFrame()
            except Exception as e:
                print(f"Error parsing holdings data: {e}")
                holdings_summary = pd.DataFrame()
                
        # Extract as-of date - default to requested date
        as_of_date = date
                
        # Try to find the actual date in the NAV summary
        def find_date_in_nav():
            for col in range(nav_summary.shape[1]):
                for row in range(nav_summary.shape[0]):
                    # Check for common labels that indicate the as-of date
                    cell_value = str(nav_summary.iloc[row, col]).lower()
                    if any(term in cell_value for term in ["as of", "holdings", "date"]):
                        # Look for a date in this row or the next row
                        for date_col in range(nav_summary.shape[1]):
                            if date_col != col:  # Skip the label column
                                try:
                                    potential_date = pd.to_datetime(nav_summary.iloc[row, date_col], errors='coerce')
                                    if pd.notna(potential_date):
                                        return potential_date
                                except:
                                    pass
                                
                                # Also try the cell below if we're not on the last row
                                if row < nav_summary.shape[0] - 1:
                                    try:
                                        potential_date = pd.to_datetime(nav_summary.iloc[row+1, date_col], errors='coerce')
                                        if pd.notna(potential_date):
                                            return potential_date
                                    except:
                                        pass
            return None
            
        try:
            extracted_date = find_date_in_nav()
            if extracted_date is not None:
                as_of_date = extracted_date
        except Exception as e:
            print(f"Warning: Could not extract as-of date from metadata: {e}")

        # Check if holdings data was successfully parsed and has content
        if holdings_summary is not None and not holdings_summary.empty:
            # Add metadata columns
            holdings_summary['date_as_of'] = as_of_date
            holdings_summary['ticker'] = etf
            
            # Clean up holdings data - remove empty columns and rows
            holdings_summary = holdings_summary.dropna(axis=1, how='all')
            holdings_summary = holdings_summary.dropna(axis=0, how='all')
        else:
            print(f"Warning: No holdings data found for {etf} on {dt_str}")
            holdings_summary = pd.DataFrame({'date_as_of': [as_of_date], 'ticker': [etf]})
        
        # Format NAV data
        nav_write = nav_summary.transpose()
        nav_write['ticker'] = etf
        
        # Clean up NAV data too
        nav_write = nav_write.dropna(axis=1, how='all')
        nav_write = nav_write.dropna(axis=0, how='all')
        
        return {
            'holdings': holdings_summary,
            'nav': nav_write
        }
    
    except Exception as e:
        print(f"Error retrieving data for {etf} on {dt_str}: {e}")
        return None

def get_historical_holdings(etf, start_date=None, end_date=None, freq='M', max_retries=3):
    """
    Get historical holdings for an ETF over a date range with retry logic
    
    Parameters:
    etf (str): ETF ticker symbol (e.g., 'IWM')
    start_date (datetime or str, optional): Start date for historical data
    end_date (datetime or str, optional): End date for historical data
    freq (str, optional): Frequency for dates ('M' for monthly, 'W' for weekly)
    max_retries (int, optional): Maximum number of retry attempts for each date
    
    Returns:
    dict: Dictionary with 'holdings' and 'nav' dataframes containing all historical data
    """
    # Set default date range
    if end_date is None:
        end_date = pd.Timestamp.now().normalize()
    else:
        end_date = pd.Timestamp(end_date)
    
    if start_date is None:
        # Default to 1 year of data
        start_date = end_date - pd.Timedelta(days=365)
    else:
        start_date = pd.Timestamp(start_date)
    
    # Generate dates based on frequency
    if freq == 'M':
        # Monthly end dates
        dates = pd.date_range(start=start_date, end=end_date, freq='BM')
    elif freq == 'W':
        # Weekly (Fridays)
        dates = pd.date_range(start=start_date, end=end_date, freq='W-FRI')
    else:
        # Custom frequency
        dates = pd.date_range(start=start_date, end=end_date, freq=freq)
    
    # Add the latest date if not included
    if end_date not in dates and len(dates) > 0:
        dates = dates.append(pd.DatetimeIndex([end_date]))
    elif len(dates) == 0:
        # If no dates in range, just use the end date
        dates = pd.DatetimeIndex([end_date])
    
    # Get holdings for each date
    holdings_list = []
    nav_list = []
    successful_dates = []
    
    for date in dates:
        date_str = date.strftime('%Y-%m-%d')
        print(f"Retrieving {etf} holdings for {date_str}...")
        
        # Try a few dates around the target date if the exact date fails
        success = False
        for attempt in range(max_retries):
            # On first attempt, use exact date
            if attempt == 0:
                try_date = date
            # On subsequent attempts, try days before
            else:
                try_date = date - pd.Timedelta(days=attempt)
            
            try_date_str = try_date.strftime('%Y-%m-%d')
            if attempt > 0:
                print(f"  Retry {attempt}/{max_retries-1}: Trying {etf} holdings for {try_date_str}...")
            
            result = get_etf_holdings(etf, try_date)
            
            if result and not result['holdings'].empty and len(result['holdings'].columns) > 2:
                print(f"  Success! Found {etf} holdings for {try_date_str}")
                holdings_list.append(result['holdings'])
                nav_list.append(result['nav'])
                successful_dates.append(try_date)
                success = True
                break
            
            # Be nice to the API server
            import time
            time.sleep(1)
        
        if not success:
            print(f"  Failed to retrieve valid holdings for {etf} around {date_str} after {max_retries} attempts")
    
    # Combine all data
    if holdings_list and nav_list:
        # Create a summary of successful retrievals
        success_summary = pd.DataFrame({
            'date_requested': dates,
            'date_retrieved': successful_dates + [None] * (len(dates) - len(successful_dates)),
            'success': [True] * len(successful_dates) + [False] * (len(dates) - len(successful_dates))
        })
        
        return {
            'holdings': pd.concat(holdings_list, ignore_index=True),
            'nav': pd.concat(nav_list, ignore_index=True),
            'retrieval_summary': success_summary
        }
    else:
        print(f"Warning: No valid holdings data found for {etf} in the specified date range")
        return None

def get_iwm_holdings(as_of_date=None, lookback_days=30):
    """
    Convenience function to get IWM holdings with improved error handling
    
    Parameters:
    as_of_date (datetime or str, optional): Date for holdings data. If None, uses current date.
    lookback_days (int, optional): Number of days to look back if the specified date fails
    
    Returns:
    DataFrame: Holdings data for IWM
    """
    if as_of_date is None:
        as_of_date = pd.Timestamp.now().normalize()
    elif isinstance(as_of_date, str):
        as_of_date = pd.Timestamp(as_of_date)
    
    # Try a series of dates, starting with the requested date and going backward
    for days_back in range(lookback_days + 1):
        try_date = as_of_date - pd.Timedelta(days=days_back)
        try_date_str = try_date.strftime('%Y-%m-%d')
        
        print(f"Attempting to retrieve IWM holdings for {try_date_str}...")
        result = get_etf_holdings('IWM', try_date)
        
        if result and not result['holdings'].empty and len(result['holdings'].columns) > 2:
            print(f"Successfully retrieved IWM holdings as of {try_date_str}")
            return result['holdings']
        
        # Wait before trying the next date
        import time
        time.sleep(1)
    
    print(f"Failed to retrieve valid IWM holdings after trying {lookback_days+1} dates")
    return None