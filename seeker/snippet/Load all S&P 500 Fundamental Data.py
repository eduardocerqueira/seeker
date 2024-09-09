#date: 2024-09-09T17:03:57Z
#url: https://api.github.com/gists/dae1fc7228a2fce3412bac023c589678
#owner: https://api.github.com/users/srkim

import pandas as pd
import yfinance as yf

def to_millions(value):
    """Convert value to millions, or return 'N/A' if not available."""
    try:
        return value / 1_000_000 if value and value != 'N/A' else 'N/A'
    except TypeError:
        return 'N/A'

def get_fundamentals(ticker):
    """Retrieve stock fundamentals for a given ticker."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
    except Exception as e:
        print(f"Error retrieving data for {ticker}: {e}")
        return None

    return {
        'Ticker': ticker,
        'Market Cap (M)': to_millions(info.get('marketCap')),
        'Enterprise Value (M)': to_millions(info.get('enterpriseValue')),
        'P/E Ratios': {
            'Trailing P/E': info.get('trailingPE', 'N/A'),
            'Forward P/E': info.get('forwardPE', 'N/A'),
            'PEG Ratio': info.get('pegRatio', 'N/A')
        },
        'Price Multiples': {
            'Price to Book': info.get('priceToBook', 'N/A'),
            'Price to Sales': info.get('priceToSalesTrailing12Months', 'N/A')
        },
        'Dividend Yield': info.get('dividendYield', 'N/A'),
        'EPS': info.get('trailingEps', 'N/A'),
        'Revenue (M)': to_millions(info.get('totalRevenue')),
        'Gross Profit (M)': to_millions(info.get('grossProfits')),
        'EBITDA (M)': to_millions(info.get('ebitda')),
        'Net Income (M)': to_millions(info.get('netIncomeToCommon')),
        'Debt to Equity': info.get('debtToEquity', 'N/A'),
        'Ratios': {
            'Current Ratio': info.get('currentRatio', 'N/A'),
            'Quick Ratio': info.get('quickRatio', 'N/A')
        },
        'Risk Metrics': {
            'Audit Risk': info.get('auditRisk', 'N/A'),
            'Board Risk': info.get('boardRisk', 'N/A'),
            'Compensation Risk': info.get('compensationRisk', 'N/A'),
            'Shareholder Rights Risk': info.get('shareHolderRightsRisk', 'N/A'),
            'Overall Risk': info.get('overallRisk', 'N/A')
        },
        'Prices': {
            'Previous Close': info.get('previousClose', 'N/A'),
            'Open': info.get('open', 'N/A'),
            'Day High': info.get('dayHigh', 'N/A'),
            'Day Low': info.get('dayLow', 'N/A'),
            '52 Week High': info.get('fiftyTwoWeekHigh', 'N/A'),
            '52 Week Low': info.get('fiftyTwoWeekLow', 'N/A')
        },
        'Volume': {
            'Current Volume': info.get('volume', 'N/A'),
            'Average Volume': info.get('averageVolume', 'N/A')
        },
        'Company Details': {
            'Name': info.get('longName', 'N/A'),
            'Exchange': info.get('exchange', 'N/A'),
            'Currency': info.get('currency', 'N/A'),
            'Financial Currency': info.get('financialCurrency', 'N/A')
        }
    }

# Load S&P 500 tickers from Wikipedia
url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
df = pd.read_html(url, header=0)[0]

# Extract and collect fundamentals for each ticker
fundamental_data = []
for ticker in df['Symbol']:
    data = get_fundamentals(ticker)
    if data:
        fundamental_data.append(data)

# Create DataFrame from collected fundamentals and save to Excel
fundamental_df = pd.DataFrame(fundamental_data)
fundamental_df.to_excel('SP500_Fundamentals.xlsx', index=False)