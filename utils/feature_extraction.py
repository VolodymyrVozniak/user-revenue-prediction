import pandas as pd
import numpy as np


def get_features_from_cumulative(df: pd.DataFrame, col_prefix: str) -> pd.DataFrame:
    sub_df : pd.DataFrame = df.filter(regex=(f"^{col_prefix}\d"))
    period_points : np.ndarray = sub_df.columns.str.extract("(\d)").values.squeeze().astype(int)
    period_len : np.ndarray = np.diff(period_points)

    #  abs cols
    abs_gain_cols : list[str] = [f'{col_prefix}_abs_gain{i}' for i in range(len(period_len))]
    abs_gain_per_period : pd.DataFrame = sub_df.diff(axis=1).iloc[:, 1:]
    abs_gain_per_period.columns = abs_gain_cols

    avg_abs_gain_cols : list[str] = [f'{col_prefix}_avg_abs_gain{i}' for i in range(len(period_len))]
    avg_abs_gain_per_period : pd.DataFrame = abs_gain_per_period / period_len
    avg_abs_gain_per_period.columns = avg_abs_gain_cols

    #  rel cols
    rel_gain_cols : list[str] = [f'{col_prefix}_rel_gain{i}' for i in range(len(period_len))]
    rel_gain_per_period : pd.DataFrame = sub_df.pct_change(axis=1).iloc[:, 1:].replace([np.inf, -np.inf, np.nan], 0)
    rel_gain_per_period.columns = rel_gain_cols

    avg_rel_gain_cols : list[str] = [f'{col_prefix}_avg_rel_gain{i}' for i in range(len(period_len))]
    avg_rel_gain_per_period : pd.DataFrame = rel_gain_per_period / period_len
    avg_rel_gain_per_period.columns = avg_rel_gain_cols

    #  merge new features in one df and return
    new_features : list(pd.DataFrame) = [abs_gain_per_period, 
                                        avg_abs_gain_per_period,
                                        rel_gain_per_period,
                                        avg_rel_gain_per_period]
    
    return pd.concat(new_features, axis=1).astype("float32")


def get_features_as_ratio(df: pd.DataFrame, numerator_col_prefix: str, denominator_col_prefix: str) -> pd.DataFrame:
    numerator_df : pd.DataFrame = df.filter(regex=(f"^{numerator_col_prefix}\d"))
    numerator_period_points : np.ndarray = numerator_df.columns.str.extract("(\d)").values.squeeze().astype(int)

    denominator_df : pd.DataFrame = df.filter(regex=(f"^{denominator_col_prefix}\d"))
    denominator_period_points : np.ndarray = denominator_df.columns.str.extract("(\d)").values.squeeze().astype(int)

    assert (numerator_period_points == denominator_period_points).all(), 'time periods are not equal'
    period_points = numerator_period_points

    new_col_names = [f'{numerator_col_prefix}_per_{denominator_col_prefix}{pp}' for pp in period_points]
    numerator_df.columns = new_col_names
    denominator_df.columns = new_col_names

    return numerator_df.div(denominator_df).replace([np.inf, -np.inf, np.nan], 0).astype("float32")
