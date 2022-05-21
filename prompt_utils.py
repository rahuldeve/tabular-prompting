import polars as pl
import numpy as np
from functools import partial


def airport_token_sequencer(name: str, num_prompts: int) -> pl.Series:
    return pl.Series([f"[{name}_{i}]" for i in range(num_prompts)])


def generate_query_prompts(
    df: pl.DataFrame, query_prompt_format: pl.Expr, num_prompts: int
) -> pl.DataFrame:
    query_prompts = df.select(
        [
            pl.all(),
            pl.col("Origin")
            .apply(partial(airport_token_sequencer, num_prompts=num_prompts))
            .arr.join(" ")
            .alias("airport_tokens"),
        ]
    ).select([query_prompt_format.alias("query")])

    return query_prompts


def generate_value_prompts(
    df: pl.DataFrame, value_prompt_format: pl.Expr
) -> pl.DataFrame:
    value_prompts = df.select(
        [
            pl.all().exclude("counts"),
            (np.floor(pl.col("counts") / 100) * 100)  # type: ignore
            .cast(pl.Int64)
            .apply(lambda x: str(x)),
        ]
    ).select([value_prompt_format.alias("value")])

    return value_prompts


def generate_query_value_prompts(
    df: pl.DataFrame, query_prompt_format: pl.Expr, value_prompt_format: pl.Expr, num_prompts: int
) -> pl.DataFrame:

    query_prompts = generate_query_prompts(df, query_prompt_format, num_prompts)
    value_prompts = generate_value_prompts(df, value_prompt_format)
    query_value_prompts = pl.concat([query_prompts, value_prompts], how="horizontal")
    return query_value_prompts
