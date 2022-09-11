import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

from sklearn.metrics import mean_absolute_error, get_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline


SCORE_NAMES = {
    "neg_root_mean_squared_error": "RMSE",
    "neg_mean_absolute_error": "MAE",
    "neg_mean_absolute_percentage_error": "MAPE",
    "r2": "R^2"
}


def plot_regression(estimator, X_train, X_test, y_train, y_test):
    """plots predicted and actual value"""

    estimator.fit(X_train, y_train)
    y_predicted = estimator.predict(X_test)
    score = mean_absolute_error(y_test, y_predicted)

    plt.figure(figsize=(15, 7))
    plt.plot(y_test, 'y')
    plt.plot(y_predicted, 'r')

    plt.axis('tight')
    model_name = re.findall(r'^[^.(]+', str(estimator))[0]
    plt.title(f"model {model_name}, mae: {score}")
    plt.grid(True)
    plt.show()

    return score


def cv_scores(model, X_train, y_train,  scoring, cv, verbose=0):
    """realizes cross_validate function for scorings"""

    scores = cross_validate(model, X_train, y_train, cv=cv, n_jobs=-1, scoring=scoring)
    
    cv_results = {}
    for score in scoring:
        cv_scores = np.mean(scores[f'test_{score}'])
        score = SCORE_NAMES[score] if score in SCORE_NAMES else score
        cv_results[f'{score}'] = cv_scores

    if verbose > 2:
        print(f"fit_time: {cv_results['fit_time']}")
        print(f"score_time: {cv_results['score_time']}\n")
        for score in scoring:
            score = SCORE_NAMES[score] if score in SCORE_NAMES else score
            print(f"mean cv {score} scores: {cv_results[f'{score}']}")

    return cv_results


def test_scores(model, X_train, X_test, y_train, y_test, scoring, verbose=0):
    """returns trained model and different scorings"""

    model.fit(X_train, y_train)

    test_results = {}
    for score in scoring:
        score_func = get_scorer(score)
        test_scores = score_func(model, X_test, y_test)

        score = SCORE_NAMES[score] if score in SCORE_NAMES else score

        test_results[f'{score}'] = test_scores

        if verbose > 2:
            print(f'test {score} scores:', test_scores)

    return model, test_results


def combine_model_results(cv, test, model_name):
    """processes and merges cv and test scores results"""

    indexs = []
    for idx in range(len(cv)):
        indexs.append(('cv', list(cv.keys())[idx]))

    index = pd.MultiIndex.from_tuples(indexs, names=["data", "metrics"])
    cv_series = pd.Series(cv.values(), index=index, name=model_name)

    indexs = []
    for idx in range(len(test)):
        indexs.append(('test', list(test.keys())[idx]))

    index = pd.MultiIndex.from_tuples(indexs, names=["data", "metrics"])
    test_series = pd.Series(test.values(), index=index, name=model_name)

    return pd.concat([cv_series, test_series])


def baseline(model, X_train, X_test, y_train, y_test, cv, scorings, verbose=0):
    """returns pretrained model and cv score with test score as series"""

    cv_result = cv_scores(deepcopy(model), X_train, y_train, scorings, cv, verbose)
    model, test_result = test_scores(model, X_train, X_test, y_train, y_test, scorings, verbose)

    model_name = re.findall(r'[\w]+\(', str(model))[-1][:-1]
    results = combine_model_results(cv_result, test_result, model_name)

    return model, results


def ml_baseline(X_train, X_test, y_train, y_test, models_info, cv, scorings, verbose=0):
    """run baseline with different models"""

    df = pd.DataFrame()

    for model, normalize in models_info:

        if normalize:
            model = make_pipeline(StandardScaler(), model)

        m, results = baseline(model=model,
                              X_train=X_train,
                              X_test=X_test,
                              y_train=y_train,
                              y_test=y_test,
                              cv=cv,
                              scorings=scorings,
                              verbose=verbose)
        df = results if df.empty else pd.concat([df, results], axis=1)

        if verbose:
            print(f'{results}\n')
        if verbose == -1:
            plot_regression(m, X_train, X_test, y_train, y_test)

    return df.T
