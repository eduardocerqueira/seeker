#date: 2025-01-01T16:35:17Z
#url: https://api.github.com/gists/50efb5a0b2e0e338a992559c643ec375
#owner: https://api.github.com/users/FinBird

import datetime
from itertools import chain, islice, groupby

def dates_between(start_year, end_year):
    """Generates dates between start_year and end_year."""
    start_date = datetime.date(start_year, 1, 1)
    end_date = datetime.date(end_year, 1, 1)
    return (start_date + datetime.timedelta(days=i) for i in range((end_date - start_date).days))

def group_by(iterable, key):
    """Groups an iterable by a key function."""
    return [list(group) for key, group in groupby(iterable, key)]

def by_month(dates):
    """Groups dates by month."""
    return group_by(dates, lambda d: d.month)

def by_week(dates):
    """Groups dates by week."""
    return group_by(dates, lambda d: (d.year, d.isocalendar()[1]))

def month_title(date):
    """Formats month title."""
    name = date.strftime("%B")
    title = " " * 22
    return title[:(22 - len(name)) // 2] + name + title[(22 + len(name)) // 2:]

def format_day(date):
    """Formats day."""
    return f"{date.day:3}"

def format_weeks(week):
    """Formats week."""
    weeks = " " * (week[0].weekday() * 3)
    weeks += "".join(format_day(d) for d in week)
    return weeks.ljust(22)

def layout_months(month):
    """Layouts month."""
    weeks = list(by_week(month))
    week_count = len(weeks)
    return [month_title(month[0]),
            " Su Mo Tu We Th Fr Sa ",
            *map(format_weeks, weeks),
            *[" " * 22] * (6 - week_count)]

def transpose(iterable):
    """Transposes a 2D iterable."""
    return list(map(list, zip(*iterable)))

def chunk(iterable, n):
    """Chunks an iterable into groups of n."""
    it = iter(iterable)
    return iter(lambda: tuple(islice(it, n)), ())


def repeat_n(value, n):
    return (value for _ in range(n))

def concat(*iterables):
    return chain(*iterables)

def join(iterable):
    return "".join(iterable)


if __name__ == "__main__":
    months = list(by_month(dates_between(2025, 2026)))
    laid_out_months = [layout_months(month) for month in months]
    chunked_months = list(chunk(laid_out_months, 3))
    transposed_chunks = [list(transpose(months)) for months in chunked_months]
    flattened_chunks = list(chain.from_iterable(transposed_chunks))
    joined_months = [join(month) for month in flattened_chunks]

    for line in joined_months:
        print(line)