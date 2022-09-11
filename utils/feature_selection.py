import logging
import os
import copy
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold


LOGS_LOCATION = "logs"
LOG_FILENAME = f"{LOGS_LOCATION}/feature_selection.log"


def remove_notunique(X: pd.DataFrame):
    """Remove features with nunique values = 1"""

    support = (X.nunique() != 1).tolist()

    columns_keep = list(X.columns[support])
    columns_drop = list(set(X.columns) - set(columns_keep))

    return columns_keep, columns_drop


def remove_variance(X: pd.DataFrame, threshold: float = .05):
    """Remove features with variance < `threshold`"""

    step_over_columns = 1000
    selector = VarianceThreshold(threshold=threshold)

    def chunker(seq, size):
        return (seq.iloc[:, pos:pos + size] for pos in range(0, seq.shape[1], size))

    support = np.array([])
    for chunk in chunker(X, step_over_columns):
        try:
            selector.fit(chunk)
            support = np.append(support, chunk.columns[selector.get_support(indices=True)])
        except ValueError as err:
            logging.error(err)

    columns_keep = list(X[support].columns)
    columns_drop = list(set(X.columns) - set(columns_keep))

    return columns_keep, columns_drop


def remove_correlated(X: pd.DataFrame, threshold: float = .95):
    """Remove one of two correlated columns > `threshold`"""

    corr = np.abs(np.corrcoef(X.T))
    bool_corr = corr >= threshold

    mask = np.tril_indices(corr.shape[0])
    bool_corr[mask] = False
    bool_columns = np.any(bool_corr, axis=0)

    columns_keep = list(X.columns[~bool_columns])
    columns_drop = list(set(X.columns) - set(columns_keep))

    return columns_keep, columns_drop


def rf_selection(X: pd.DataFrame, y: np.ndarray, threshold : float = .95, seed: int = None):
    """
    Select features by RandomForestRegressor
    with feature importance > `threshold` in total
    """

    sorted_imp_categ_copy = []

    if X.any().any():
        rf = RandomForestRegressor(max_features='auto', random_state=seed, n_jobs=-1)
        rf.fit(X, y)

        if not rf.feature_importances_.any():
            return X.columns, [], None
        
        sorted_imp_categ = sorted(zip(X.columns, rf.feature_importances_),
                                  key=lambda x: x[1], reverse=True)
        sorted_imp_categ_copy = copy.deepcopy(sorted_imp_categ)

        summ = 1.
        while summ > threshold:
            summ -= sorted_imp_categ_copy.pop()[1]

    columns_keep = [sorted_imp_categ_copy[idx][0] for idx in range(len(sorted_imp_categ_copy))]
    columns_drop = list(set(X.columns) - set(columns_keep))

    return columns_keep, columns_drop, sorted_imp_categ


def select_features(X: pd.DataFrame, y: np.ndarray = None, seed: int = None) -> np.ndarray:
    """
    Remove with nunique = 1 
    Remove with small variance
    Remove with high correlation
    Remove with small feature importance by RF
    """

    if not os.path.exists(LOGS_LOCATION):
        os.makedirs(LOGS_LOCATION)

    logging.basicConfig(filename=LOG_FILENAME, filemode='w', level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(message)s')

    assert X.shape[1] == X.select_dtypes(include='number').shape[1]
    if X.empty:
        return np.empty(shape=0, dtype=str)

    X = X.copy()
    var_threshold = .05
    corr_threshold = .95
    rf_threshold = .95

    print(f'Columns: {len(X.columns)}')
    logging.info(f"Columns: {len(X.columns)}")

    columns_keep, columns_drop = remove_notunique(X)
    print(f'Columns after not unique: {len(columns_keep)}')
    logging.info(f"Columns after not unique: {len(columns_keep)}")
    logging.info(f"Columns dropped: {columns_drop}")

    columns_keep, columns_drop = remove_variance(X[columns_keep], var_threshold)
    print(f'Columns after var: {len(columns_keep)}')
    logging.info(f"Columns after var: {len(columns_keep)}")
    logging.info(f"Columns dropped: {columns_drop}")

    columns_keep, columns_drop = remove_correlated(X[columns_keep], corr_threshold)
    print(f'Columns after corr: {len(columns_keep)}')
    logging.info(f"Columns after corr: {len(columns_keep)}")
    logging.info(f"Columns dropped: {columns_drop}")

    if y is not None:
        columns_keep, columns_drop, _ = rf_selection(X[columns_keep], y, rf_threshold, seed)
        print(f'Columns after RF: {len(columns_keep)}')
        logging.info(f"Columns after RF: {len(columns_keep)}")
        logging.info(f"Columns dropped: {columns_drop}")

    logging.info(f"Columns kept: {columns_keep}")

    return np.array(columns_keep, dtype=str)
