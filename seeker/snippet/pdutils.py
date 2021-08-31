#date: 2021-08-31T13:19:34Z
#url: https://api.github.com/gists/0f9c5971a62b5e08a21b71f15aec525a
#owner: https://api.github.com/users/sgraaf

import csv
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd
from tqdm import tqdm


def read_csv(
    file: Path,
    sep: Optional[str] = ",",
    header: Optional[Union[List[int], int]] = "infer",
    usecols: Optional[List[int]] = None,
    names: Optional[List[str]] = None,
    encoding: Optional[str] = "utf-8",
) -> pd.DataFrame:
    if not (usecols is None and names is None):
        assert len(usecols) == len(names)
    return pd.read_csv(
        file, sep=sep, header=header, usecols=usecols, names=names, encoding=encoding
    )


def read_csvs(
    files: List[Path],
    sep: Optional[str] = ",",
    header: Optional[Union[List[int], int]] = "infer",
    usecols: Optional[List[int]] = None,
    names: Optional[List[str]] = None,
    encoding: Optional[str] = "utf-8",
    use_tqdm: Optional[bool] = False,
) -> pd.DataFrame:
    if not (usecols is None and names is None):
        assert len(usecols) == len(names)

    if use_tqdm:
        files = tqdm(files, desc="Reading CSV-files", total=len(files), unit="file")
    return pd.concat(
        [
            read_csv(
                file,
                sep=sep,
                header=header,
                usecols=usecols,
                names=names,
                encoding=encoding,
            )
            for file in files
        ]
    )


def read_xlsx(
    file: Path,
    names: Optional[List[str]] = None,
    usecols: Optional[List[Union[int, str]]] = None,
) -> pd.DataFrame:
    return pd.read_excel(file, names=names, usecols=usecols)


def read_xlsxs(
    files: List[Path],
    names: Optional[List[str]] = None,
    usecols: Optional[List[Union[int, str]]] = None,
    use_tqdm: Optional[bool] = False,
) -> pd.DataFrame:
    if use_tqdm:
        files = tqdm(files, desc="Reading XLSX-files", total=len(files), unit="file")
    return pd.concat([read_xlsx(file, names=names, usecols=usecols) for file in files])


def write_csv(
    df: pd.DataFrame,
    file: Path,
    sep: Optional[str] = ",",
    encoding: Optional[str] = "utf-8",
    quoting: Optional[int] = csv.QUOTE_MINIMAL,
    quotechar: Optional[str] = '"',
    decimal: Optional[str] = ".",
) -> None:
    df.to_csv(
        file,
        sep=sep,
        index=False,
        encoding=encoding,
        quoting=quoting,
        quotechar=quotechar,
        decimal=decimal,
    )


def write_excel(
    df: pd.DataFrame, file: Path, engine: Optional[str] = "openpyxl"
) -> None:
    df.to_excel(file, index=False, engine=engine)


def sort_rows(
    df: pd.DataFrame, by: Union[List[str], str], ascending: Optional[bool] = True
) -> pd.DataFrame:
    return df.sort_values(by=by, ascending=ascending)
