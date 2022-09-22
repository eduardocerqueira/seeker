#date: 2022-09-22T17:05:20Z
#url: https://api.github.com/gists/19d3f511c4694c266c373cfb653740a9
#owner: https://api.github.com/users/NChechulin

from datetime import datetime
from enum import Enum, auto
from io import StringIO
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd
import numpy as np
import seaborn as sns
from dateutil.relativedelta import relativedelta

YAHOO_BASE_URL = "https://query1.finance.yahoo.com/v7/finance/download/"
sns.set(rc={"figure.figsize": (18, 12)})  # type: ignore


class TimePeriod(Enum):
    DAY = auto()
    FIVE_DAYS = auto()
    THREE_MONTHS = auto()
    SIX_MONTHS = auto()
    YTD = auto()
    YEAR = auto()
    FIVE_YEARS = auto()
    ALL = auto()

    def get_timestamps(self) -> tuple[int, int]:
        end = datetime.now()
        start: datetime = None  # type: ignore

        match self:
            case TimePeriod.DAY:
                start = end - relativedelta(days=1)
            case TimePeriod.FIVE_DAYS:
                start = end - relativedelta(days=5)
            case TimePeriod.THREE_MONTHS:
                start = end - relativedelta(months=3)
            case TimePeriod.SIX_MONTHS:
                start = end - relativedelta(months=6)
            case TimePeriod.YTD:
                start = datetime(year=end.year, month=1, day=1)
            case TimePeriod.YEAR:
                start = end - relativedelta(years=1)
            case TimePeriod.FIVE_YEARS:
                start = end - relativedelta(years=5)
            case TimePeriod.ALL:
                start = datetime(year=1800, month=1, day=1)

        return (
            int(start.timestamp()),
            int(end.timestamp()),
        )


class DataFrameColumn:
    DATE = "Date"
    OPEN = "Open"
    HIGH = "High"
    LOW = "Low"
    CLOSE = "Close"
    ADJ_CLOSE = "Adj Close"
    VOLUME = "Volume"


class Stock:
    __ticker: str
    __df: pd.DataFrame = None  # type: ignore

    def __init__(self, ticker: str) -> None:
        self.__ticker = ticker

    def download(self, period: TimePeriod) -> None:
        """
        Downloads the data from Yahoo Finance in a specified date range.

        Parameters
        ----------
        period : TimePeriod
        """
        start, end = period.get_timestamps()

        payload = urlencode(
            {
                "interval": "1d",
                "period1": start,
                "period2": end,
                "events": "history",
                "includeAdjustedClose": "true",
            }
        )

        csv_text = ""

        url = YAHOO_BASE_URL + self.__ticker + f"?{payload}"
        with urlopen(url) as response:
            csv_text = response.read().decode()

        csv = StringIO(csv_text)
        self.__df = pd.read_csv(csv)  # type: ignore
        self.__cast_date()

    def __cast_date(self) -> None:
        """
        Replaces date string with actual date in the dataframe.
        """

        date = DataFrameColumn.DATE
        self.__df[date] = pd.to_datetime(self.__df[date])  # type: ignore

    @property
    def dataframe(self) -> pd.DataFrame:
        """
        Returns the dataframe reference.

        Returns
        -------
        pd.DataFrame
        """
        return self.__df

    def add_column(self, name: str, column: pd.Series) -> None:  # type: ignore
        self.__df[name] = column

    def calculate_adjusted(self, column_name: str) -> pd.Series:  # type: ignore
        """
        Calculates adjusted

        Parameters
        ----------
        column_name : str
            Name of the column (better be passed via `DataFrameColumn`)

        Returns
        -------
        pd.Series
            Column with adjusted data
        """

        if column_name not in [
            DataFrameColumn.CLOSE,
            DataFrameColumn.OPEN,
            DataFrameColumn.LOW,
            DataFrameColumn.HIGH,
        ]:
            raise ValueError("Wrong column specified")

        COLS = DataFrameColumn
        adj_coeff = self.__df[COLS.ADJ_CLOSE] / self.__df[COLS.CLOSE]
        return self.__df[column_name] * adj_coeff  # type: ignore

    def adjust_inplace(self, replace_data: bool = False) -> None:
        """
        Calculates adjusted values of all suitable columns.

        Parameters
        ----------
        replace_data : bool, optional
            If set to True, replace the data without creating new columns.
            If set to False, creates 4 'Adjusted' columns. By default False.
        """
        for column_name in [
            DataFrameColumn.CLOSE,
            DataFrameColumn.OPEN,
            DataFrameColumn.LOW,
            DataFrameColumn.HIGH,
        ]:
            new_name = column_name if replace_data else f"Adj {column_name}"
            self.__df[new_name] = self.calculate_adjusted(column_name)  # type: ignore
        if replace_data:
            self.__df.drop(columns=DataFrameColumn.ADJ_CLOSE)

    def calculate_returns(self) -> np.array:
        colname = DataFrameColumn.ADJ_CLOSE
        if DataFrameColumn.ADJ_CLOSE not in self.__df.columns:
            colname = DataFrameColumn.CLOSE

        upper: np.array = np.array(self.__df[colname][1:])  # type: ignore
        lower: np.array = np.array(self.__df[colname][:-1])  # type: ignore
        return (upper / lower) - 1  # type: ignore

    def plot(self, column: str) -> None:
        """
        Plots a given columns on a line plot.

        Parameters
        ----------
        column : str
        """
        sns.lineplot(
            x=DataFrameColumn.DATE,
            y=column,
            data=self.__df,
        )
