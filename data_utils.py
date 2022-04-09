from glob import glob
from typing import List, Sequence, cast
import polars as pl


def load_data(start_year: int, end_year: int) -> pl.DataFrame:
    files = glob("dataverse_files/[0-9]*.csv")
    df = [pl.scan_csv(f, null_values="NA") for f in files]  # type: ignore
    df = pl.concat(df, how="horizontal")  # type: ignore

    df: pl.DataFrame = df.select(
        ["Year", "Month", "DayofMonth", "DayOfWeek", "Origin"]
    ).filter((pl.col("Year") > start_year) & (pl.col('Year') < end_year)).groupby(
        ["Year", "Month", "DayofMonth", "DayOfWeek", "Origin"]
    ).agg(
        pl.count().alias("counts")
    ).collect()

    # months_df = pl.DataFrame(
    #     {
    #         "Month": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    #         "Month_name": [
    #             "January",
    #             "February",
    #             "March",
    #             "April",
    #             "May",
    #             "June",
    #             "July",
    #             "August",
    #             "September",
    #             "October",
    #             "November",
    #             "December",
    #         ],
    #     }
    # )

    airports_df = pl.read_csv("dataverse_files/airports.csv")
    airports_df = airports_df.select([pl.col("iata"), pl.col("airport")])

    df = df.join(airports_df, left_on="Origin", right_on="iata")
    # df = df.join(months_df, on="Month")
    return df
