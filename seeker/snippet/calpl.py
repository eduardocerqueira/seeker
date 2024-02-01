#date: 2024-02-01T17:09:41Z
#url: https://api.github.com/gists/81512b119e26e2c4a5b235989653fec2
#owner: https://api.github.com/users/deanm0000

def pl_cal_sheet(
    wb: CalamineWorkbook,
    sheet: str,
    header_rows: int | None = None,
    header_merge_char: str = "_",
    skip_rows: int = 0,
    infer_schema_length: int = 1000,
    infer_schema_minrow: int = 10,
    column_dupe_name_seperator: str = "_",
) -> pl.DataFrame:
    """
    Read sheet from CalamineWorkbook into a DataFrame

    Parameters
    ----------
    wb
        A CalamineWorkbook object
    sheet
        The name of a sheet in the Workbook
    header_rows
        The number of rows that make up a distinct column. If this is None, it will
        be inferred based on non-null inferred dtypes.
    skip_rows
        The number of rows to skip. Row skipping happens before anything else
    infer_schema_length
        The number of rows to load to try casting to Int64, Float64, Date
    infer_schema_minrow
        The first row to consider when attempting to infer dtypes
    column_dupe_name_seperator
        If there are duplicate column names, the duplicates will have a suffix
        added where the suffix is this seperator followed by the index for example
        col, col_1, col_2, etc
    """
    df = pl.from_records(wb.get_sheet_by_name(sheet).to_python(), orient="row").slice(
        skip_rows
    )
    if (
        header_rows is not None
        and isinstance(header_rows, int)
        and header_rows > infer_schema_minrow
    ):
        infer_schema_minrow = header_rows + 1
    df_typing = df.slice(infer_schema_minrow, infer_schema_length)
    for col_name in df_typing.columns:
        for dtype in [pl.Int64, pl.Float64, pl.Date]:
            try:
                df_typing = df_typing.with_columns(
                    pl.col(col_name).replace({"": None}).cast(dtype)
                )
                continue
            except:  # noqa: E722
                pass
    non_strings = [
        (col_name, dtype)
        for col_name, dtype in df_typing.schema.items()
        if dtype != pl.String
    ]
    if header_rows is None and len(non_strings) > 0:
        header_rows = (
            df.slice(0, infer_schema_minrow)
            .select(pl.col(x[0]).cast(x[1], strict=False) for x in non_strings)
            .with_row_index("__i")
            .select(
                pl.min_horizontal(
                    pl.col("__i").filter(pl.col(x[0]).is_not_null()).min().alias(x[0])
                    for x in non_strings
                )
            )
            .item()
        )
    elif header_rows is None and len(non_strings) == 0:
        header_rows = 1

    df_cols = (
        pl.select(
            pl.Series(
                "colnames",
                [
                    header_merge_char.join([y for y in x if y != ""])
                    for x in zip(
                        *[df.slice(x, 1).rows()[0] for x in range(header_rows)]
                    )
                ],
            )
        )
        .with_columns(count=pl.int_range(0, pl.len()).over("colnames"))
        .with_columns(
            colnames=pl.when(pl.col("count") > 0)
            .then(
                pl.col("colnames")
                + pl.lit(column_dupe_name_seperator)
                + pl.col("count").cast(pl.Utf8)
            )
            .otherwise(pl.col("colnames"))
        )
        .get_column("colnames")
    )

    df = df.slice(header_rows)
    df = df.with_columns(pl.col(x[0]).cast(x[1]) for x in non_strings)
    df.columns = df_cols
    return df
