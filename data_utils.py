from glob import glob
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

    airports_df = pl.read_csv("dataverse_files/airports.csv")
    airports_df = airports_df.select([pl.col("iata"), pl.col("airport")])

    df = df.join(airports_df, left_on="Origin", right_on="iata")
    return df
