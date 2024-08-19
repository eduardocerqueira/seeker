#date: 2024-08-19T17:07:24Z
#url: https://api.github.com/gists/02a0edc5610e985251c82dd4c940a278
#owner: https://api.github.com/users/davidmezzetti

import json
import re

import yfinance as yf

from txtai import Embeddings
from txtai.pipeline import Textractor

def djia():
    """
    Gets a list of stocks on the Dow Jones Industrial Average.

    Returns:
        Dow Jones Industrial Average stocks
    """

    stocks, textractor = [], Textractor(sections=True)
    for section in textractor("https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average"):
        if section.startswith("## Components"):
            for line in section.split("\n")[4:]:
                if line.count("|") > 1 and "---" not in line:
                    stock = line.split("|")[3]
                    stocks.append(re.sub(r"\[(.+?)\].*", r"\1", stock))

    return stocks

def nasdaq100():
    """
    Gets a list of stocks on the Nasdaq-100.

    Returns:
        Nasdaq-100 stocks
    """

    stocks, textractor = [], Textractor(sections=True)
    for section in textractor("https://en.wikipedia.org/wiki/Nasdaq-100"):
        if section.startswith("## Components"):
            for line in section.split("\n")[4:]:
                if line.count("|") > 1:
                    stocks.append(line.split("|")[2])

    return stocks

def sp500():
    """
    Gets a list of stocks on the S&P 500.

    Returns:
        S&P 500 stocks
    """

    stocks, textractor = [], Textractor(sections=True)
    for section in textractor("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"):
        if "S&P 500 component stocks" in section:
            for line in section.split("\n")[4:]:
                if line.count("|") > 1:
                    stock = line.split("|")[1]
                    stocks.append(re.sub(r"\[(.+?)\].*", r"\1", stock))

    return stocks

def stream():
    """
    Yields a list of unique stocks from the Dow Jones Industrial Average, Nasdaq-100 and S&P 500.

    Returns:
        stock infos
    """

    # Get stocks for target indexes
    dji, ndx, spx = djia(), nasdaq100(), sp500()

    # Get a list of unique stocks from Nasdaq-100 and S&P 500
    stocks = sorted(set(dji + ndx + spx))

    # Retrieve stock info from yfinance API
    tickers = yf.Tickers(" ".join(stocks))
    for stock in stocks:
        # Get stock info
        data = tickers.tickers[stock].info

        # Get stock indexes
        data["stockIndexes"] = [
            index for x, index in enumerate(["Dow Jones Industrial Average", "Nasdaq-100", "S&P 500"])
            if stock in [dji, ndx, spx][x]
        ]

        # Index JSON representation.
        data["text"] = json.dumps(data)

        yield data.get("shortName", stock), data

# Build embeddings index
embeddings = Embeddings(
    path="intfloat/e5-large",
    instructions={"query": "query: ", "data": "passage: "},
    content=True,
    graph={"approximate": False, "minscore": 0.7},
)
embeddings.index(stream())
embeddings.save("txtai-stocks")
