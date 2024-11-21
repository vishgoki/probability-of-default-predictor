import pickle
import pandas as pd
import numpy as np
from pygam import LogisticGAM, s

from preprocessor import preprocessor


def prediction(holdout_data_path, model_path, output):
    data = pd.read_csv(holdout_data_path)

    test_data = preprocessor(data, isTraining=True)


    variables = ["roa", "operating_profit_margin", "ocf_ratio",
                 "cash_ratio", "asset_coverage_ratio", "asset_turnover_ratio",
                 "liability_to_asset", "dscr", "tie"]

    ratio_to_quantiles = {}
    transformation_functions = {}

    for var in variables:
        try:
            with open(f'ratio_to_quantile_func_{var}.pkl', 'rb') as f:
                ratio_to_quantiles[var] = pickle.load(f)
            with open(f'transformation_function_{var}.pkl', 'rb') as f:
                transformation_functions[var] = pickle.load(f)
        except FileNotFoundError as e:
            print(f"Transformation functions for '{var}' not found: {e}")
            continue

    transformed_variables = []
    for var in variables:
        transformed_var = f'transformed_{var}'
        if transformed_var not in transformed_variables:
            transformed_variables.append(transformed_var)

    median_transformed_dict = {'transformed_roa': 0.007046658883200175,
                               'transformed_operating_profit_margin': 0.007409418174229729,
                               'transformed_ocf_ratio': 0.006276966651106275,
                               'transformed_cash_ratio': 0.01314800982092483,
                               'transformed_asset_coverage_ratio': 0.007627000624239938,
                               'transformed_asset_turnover_ratio': 0.012532754529477975,
                               'transformed_liability_to_asset': 0.007016577459023089,
                               'transformed_dscr': 0.007683202805369696,
                               'transformed_tie': 0.009161721037544278}

    for var in variables:
        print(f"Applying transformation for {var} on test data")
        RTQ = ratio_to_quantiles.get(var)
        TF = transformation_functions.get(var)
        if RTQ is None or TF is None:
            print(f"Skipping {var} due to missing transformation functions.")
            continue

        # Map ratio values to quantile positions
        quantile_positions = RTQ(test_data[var].values)
        # Apply the transformation function
        transformed_values = TF(quantile_positions)
        test_data[f'transformed_{var}'] = transformed_values

    X_test = test_data[transformed_variables].reset_index(drop=True)

    # Handle missing values in test_data[var]
    for col in X_test.columns:
        if col in median_transformed_dict.keys():
            X_test = X_test.copy()
            X_test.loc[:, col] = X_test[col].fillna(median_transformed_dict[col])

    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)

    y_pred_prob = model.predict_proba(X_test)

    S = 0.013556299997732462
    T = 0.008446501004638033

    def bayesian_adjustment(p, S, T):
        numerator = T * (p - p * S)
        denominator = S - p * S + p * (T - S * T)
        return np.clip(numerator / denominator, 0, 1)

    pd.DataFrame(y_pred_prob).to_csv("out1.csv")

    y_prob_bayesian = bayesian_adjustment(y_pred_prob, S, T)


    y_prob_bayesian = pd.DataFrame(y_prob_bayesian)
    y_prob_bayesian.to_csv(output, header=False, index=False)
    print(f"Predictions saved to {output}")
