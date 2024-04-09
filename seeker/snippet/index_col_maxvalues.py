#date: 2024-04-09T16:51:14Z
#url: https://api.github.com/gists/0bdf07ff205ad4d28e577792d10f9bd7
#owner: https://api.github.com/users/nilotpalc

import polars as pl

df = pl.DataFrame(
    {
        "a": [20, 10, 30],"b":[30.23,50]
    }
)

def arg_max_horizontal(*columns: pl.Expr) -> pl.Expr:
    return (
        pl.concat_list(columns)
        .list.arg_max()
        .replace({i: col_name for i, col_name in enumerate(columns)})
    )

# Example usage:
df = df.with_columns(Largest=arg_max_horizontal("a", "b"))