import pandas as pd
from xgboost import XGBRegressor


COLUMNS_TO_DROP = ['media_source', 'country_code',
                   'install_date', 'platform']


class UserRevenuePredictor():
    def __init__(self, bias_target: bool = False, models_location: str = "models"):
        model_name = "xgboost_bias_final" if bias_target else "xgboost_final"
        model_path = f"{models_location}/{model_name}.json"

        self.bias_target = bias_target
        self.model = XGBRegressor()
        self.model.load_model(model_path)

    def predict(self, X: pd.DataFrame):
        X = X.copy()
        X.drop(columns=COLUMNS_TO_DROP, inplace=True)
        
        pred = self.model.predict(X)

        if self.bias_target:
            pred -= 1

        return pred
