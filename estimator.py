#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from pygam import LogisticGAM, s
from sklearn.metrics import roc_curve, auc
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.interpolate import interp1d
from sklearn.tree import DecisionTreeClassifier
import os
sns.set(style="whitegrid")


# In[2]:


def preprocess_data(data):
    #### Preprocess individual variables with the financial knowledge

    data['stmt_date'] = pd.to_datetime(data['stmt_date'], format='%Y-%m-%d')
    data['def_date'] = pd.to_datetime(data['def_date'], format='%d/%m/%Y')

    def assign_default(row):
        date_diff = (row['def_date'] - row['stmt_date']).days
        return 1 if 150 <= date_diff <= 515 else 0

    data['default'] = data.apply(assign_default, axis=1)

    data['prof_operations'] = np.where(
        data['prof_operations'].isnull() & data['rev_operating'].notnull() & data['COGS'].notnull(),
        data['rev_operating'] - data['COGS'],
        data['prof_operations'])

    data['prof_operations'] = np.where(data['prof_operations'].isnull() & data['roa'].notnull() & data['asst_tot'].notnull(),
        (data['roa'] / 100) * data['asst_tot'],
        data['prof_operations'])

    data['rev_operating'] = np.where(
        data['rev_operating'].isnull() & data['prof_operations'].notnull() & data['COGS'].notnull(),
        data['prof_operations'] + data['COGS'],
        data['rev_operating'])

    data['COGS'] = np.where(
        data['COGS'].isnull() & data['rev_operating'].notnull() & data['prof_operations'].notnull(),
        data['rev_operating'] - data['prof_operations'],
        data['COGS'])

    data['roa'] = np.where(
        data['roa'].isnull() & data['prof_operations'].notnull() & data['asst_tot'].notnull(),
        data['prof_operations'] * 100 / data['asst_tot'],
        data['roa'])

    data['roe'] = np.where(
        data['roe'].isnull() & data['profit'].notnull() & data['eqty_tot'].notnull(),
        data['profit'] * 100 / data['eqty_tot'],
        data['roe'])

    data["interest_expense_after_taxes"] = data["prof_operations"] - data["profit"]

    data['interest_expense_after_taxes'] = np.where(
        data['interest_expense_after_taxes'].isnull() & data['prof_operations'].notnull() & data['profit'].notnull(),
        data['prof_operations'] - data['profit'],
        data['interest_expense_after_taxes'])

    data['asst_tot'] = np.where(
        data['asst_tot'].isnull() & data['debt_st'].notnull() & data['debt_lt'].notnull() & data['liab_lt'].notnull() & data['liab_lt_emp'].notnull() & data['eqty_tot'].notnull(),
        data['debt_st'] + data['debt_lt'] + data['liab_lt'] + data['liab_lt_emp'] + data['eqty_tot'],
        data['asst_tot'])

    data['asst_tot'] = np.where(
        data['asst_tot'].isnull() & data['asst_current'].notnull() & data['asst_intang_fixed'].notnull() & data['asst_tang_fixed'].notnull() & data['asst_fixed_fin'].notnull(),
        data['asst_current'] + data['asst_intang_fixed'] + data['asst_tang_fixed'] + data['asst_fixed_fin'],
        data['asst_tot'])

    data['debt_st'] = np.where(
        data['debt_st'].isnull() & data['asst_current'].notnull() & data['wc_net'].notnull(),
        data['asst_current'] - data['wc_net'],
        data['debt_st'])

    data['debt_st'] = np.where(
        data['debt_st'].isnull() & data['asst_tot'].notnull() & data['debt_lt'].notnull() & data['eqty_tot'].notnull() & data['liab_lt'].notnull() & data['liab_lt_emp'].notnull(),
        data['asst_tot'] - (data['debt_lt'] + data['eqty_tot'] + data['liab_lt'] + data['liab_lt_emp']),
        data['debt_st']
    )

    data['debt_lt'] = np.where(
        data['debt_lt'].isnull() & data['asst_tot'].notnull() & data['debt_st'].notnull() & data['eqty_tot'].notnull() & data['liab_lt'].notnull() & data['liab_lt_emp'].notnull(),
        data['asst_tot'] - (data['debt_st'] + data['eqty_tot'] + data['liab_lt'] + data['liab_lt_emp']),
        data['debt_lt']
    )

    data['eqty_tot'] = np.where(
        data['eqty_tot'].isnull() & data['asst_tot'].notnull() & data['debt_st'].notnull() & data['debt_lt'].notnull() & data['liab_lt'].notnull() & data['liab_lt_emp'].notnull(),
        data['asst_tot'] - (data['debt_st'] + data['debt_lt'] + data['liab_lt'] + data['liab_lt_emp']),
        data['eqty_tot']
    )

    data['liab_lt'] = np.where(
        data['liab_lt'].isnull() & data['asst_tot'].notnull() & data['debt_st'].notnull() & data['debt_lt'].notnull() & data['eqty_tot'].notnull() & data['liab_lt_emp'].notnull(),
        data['asst_tot'] - (data['debt_st'] + data['debt_lt'] + data['eqty_tot'] + data['liab_lt_emp']),
        data['liab_lt']
    )

    data['liab_lt_emp'] = np.where(
        data['liab_lt_emp'].isnull() & data['asst_tot'].notnull() & data['debt_st'].notnull() & data['debt_lt'].notnull() & data['eqty_tot'].notnull() & data['liab_lt'].notnull(),
        data['asst_tot'] - (data['debt_st'] + data['debt_lt'] + data['eqty_tot'] + data['liab_lt']),
        data['liab_lt_emp']
    )


    #### RATIOS (Feature Engineering)

    ### Profitability Ratios

    # ROA
    #data['roa']

    # Operating profit margin
    data['operating_profit_margin'] = np.where(
        (data['rev_operating'] == 0),
        0,
        data['prof_operations'] / data['rev_operating'])

    ### Liquidity Ratios

    # Operating cash flow
    data['ocf_ratio'] = np.where(
        (data['cf_operations'] == 0) & (data['debt_st'] == 0),
        0,
        data['cf_operations'] / data['debt_st']
        )

    data['ocf_ratio'] = np.where(data['ocf_ratio'] == np.inf, 100, data['ocf_ratio'])
    data['ocf_ratio'] = np.where(data['ocf_ratio'] == -np.inf, -100, data['ocf_ratio'])

    # Cash Ratio
    data['cash_ratio'] = np.where(
        (data['cash_and_equiv'] == 0) & (data['debt_st'] == 0),
        0,
        data['cash_and_equiv'] / data['debt_st']
        )

    data['cash_ratio'] = np.where(data['cash_ratio'] == np.inf, 100, data['cash_ratio'])
    data['cash_ratio'] = np.where(data['cash_ratio'] == -np.inf, -100, data['cash_ratio'])


    ### Efficiency Ratios

    # Asset turnover ratio
    data['asset_turnover_ratio'] = np.where(
        (data['asst_tot'] == 0),
        0,
        data['rev_operating'] / data['asst_tot']
        )

    ### Coverage Ratios

    # Asset coverage ratio

    data['asset_coverage_ratio'] = np.where(
        ((data['asst_tot'] - data['asst_intang_fixed']) == 0) & ((data['asst_tot'] - data['eqty_tot']) == 0),
        0,
        (data['asst_tot'] - data['asst_intang_fixed']) / (data['asst_tot'] - data['eqty_tot'])
      )

    data['asset_coverage_ratio'] = np.where(data['asset_coverage_ratio'] == np.inf, 100, data['asset_coverage_ratio'])
    data['asset_coverage_ratio'] = np.where(data['asset_coverage_ratio'] == -np.inf, -10600000, data['asset_coverage_ratio'])

    ### Leverage Ratios

    # Liability to asset
    data['liability_to_asset'] = np.where(
        (data['asst_tot'] == 0),
        0,
        (data['asst_tot'] - data['eqty_tot']) / data['asst_tot']
    )

    # Debt Service Coverage Ratio (Coverage of Debt)
    data['dscr'] = np.where(
        (data['prof_operations'] == 0) & (data['debt_st'] == 0),
        0,
        data['prof_operations'] / data['debt_st']
    )

    data['dscr'] = np.where(data['dscr'] == np.inf, 100, data['dscr'])
    data['dscr'] = np.where(data['dscr'] == -np.inf, -100, data['dscr'])

    data['roe'] = np.where(data['roe'] == np.inf, -4000, data['roe'])
    data['roe'] = np.where(data['roe'] == -np.inf, -4000, data['roe'])

    # Times interest earned

    data['tie'] = data['prof_operations'] / data["interest_expense_after_taxes"]

    data['tie'] = np.where(data['tie'] == np.inf, 100, data['tie'])
    data['tie'] = np.where(data['tie'] == -np.inf, -100, data['tie'])

    variables = ["roa", "roe", "operating_profit_margin", "ocf_ratio",
                 "cash_ratio", "asset_coverage_ratio", "asset_turnover_ratio",
                 "liability_to_asset", "dscr", "tie"]

    ### Infinities
    #data = data.replace([np.inf, -np.inf], 0)
    print("Number of infinities in selected variables:", data[variables].apply(lambda x: np.isinf(x).sum()).sum())

    median_dict = {'roa': 2.39,
                 'roe': 2.62,
                 'operating_profit_margin': 0.04007278467033522,
                 'ocf_ratio': 0.050586252780393054,
                 'cash_ratio': 0.035357562959052305,
                 'asset_coverage_ratio': 1.2027110053516417,
                 'asset_turnover_ratio': 0.5739481212885005,
                 'liability_to_asset': 0.8111305589726827,
                 'dscr': 0.055279427393172736,
                 'tie': 1.1348268165660909}



    variables = variables + ['id', 'fs_year', 'default']

    data = data[variables]
    for col in data.columns:
        if col in median_dict.keys():
            lower_bound = data[col].quantile(0.01)
            upper_bound = data[col].quantile(0.99)
            data[col] = data[col].clip(lower=lower_bound, upper=upper_bound)
            data[col] = data[col].fillna(median_dict[col])

    print("Null values filled with medians")
    return data[variables]


# In[3]:


data = pd.read_csv('train.csv')


# In[4]:


df = preprocess_data(data)


# In[5]:


def transform_ratio(data, ratio_column, default_column='default', k=51, frac=0.1, save_functions=True, functions_dir='functions/'):
    
    if save_functions and not os.path.exists(functions_dir):
        os.makedirs(functions_dir)
    data = data.copy()
    data['original_index'] = data.index


    df_sorted = data.sort_values(by=ratio_column).reset_index(drop=True)
        
        
        
    while k > 16:
        try:
            quantile_labels = [f'Q{i}' for i in range(1, k + 1)]
            df_sorted['Quantile'] = pd.qcut(df_sorted[ratio_column], q=k, labels=quantile_labels, duplicates='drop')
            break
        except ValueError as e:
            print(f"Error in qcut: {e} for bins {k}")
            k -= 5
            

    quantile_default_rates = df_sorted.groupby('Quantile', observed=True)[default_column].mean().reset_index()
    quantile_default_rates['Quantile_Num'] = quantile_default_rates['Quantile'].str.extract(r'Q(\d+)').astype(int)

    #Prepare data for LOESS smoothing
    quantile_positions = quantile_default_rates['Quantile_Num'] / k
    default_rates = quantile_default_rates[default_column]

    # Apply LOESS 
    smoothed = lowess(
        endog=default_rates,
        exog=quantile_positions,
        frac=frac,
        return_sorted=True
    )


    smoothed_quantiles = smoothed[:, 0]
    smoothed_defaults = smoothed[:, 1]


    #Compute the empirical CDF
    N = len(df_sorted)
    sorted_ratios = df_sorted[ratio_column].values
    cdf_values = np.arange(1, N + 1) / N

    cdf_df = pd.DataFrame({
        'roa': sorted_ratios,
        'cdf': cdf_values
    })

    # Aggregate CDF values for duplicate roa
    unique_cdf_df = cdf_df.groupby('roa', as_index=False)['cdf'].mean()

    # Ensure sorted_ratios are unique
    unique_sorted_ratios = unique_cdf_df['roa'].values
    unique_cdf_values = unique_cdf_df['cdf'].values

    # Create the ratio to quantile interpolation function
    ratio_to_quantile_func = interp1d(
        unique_sorted_ratios,
        unique_cdf_values,
        kind='linear',
        fill_value='extrapolate',
        bounds_error=False
    )

    # Create the transformation interpolation function using smoothed data
    transformation_function = interp1d(
        smoothed_quantiles,
        smoothed_defaults,
        kind='linear',
        fill_value=(smoothed_defaults[0], smoothed_defaults[-1]),
        bounds_error=False
    )

    # Plot the smoothed curve along with the original data
    # plt.figure(figsize=(10, 6))
    # plt.plot(smoothed_quantiles * k, smoothed_defaults, color='red', label='Smoothed Default Rate')
    # plt.scatter(quantile_default_rates['Quantile_Num'], quantile_default_rates['default'], color='blue', label='Original Default Rate')
    # plt.xlabel('Quantile')
    # plt.ylabel('Default Rate')
    # plt.title(f'Smoothed Default Rate by Quantile of {ratio_column}')
    # plt.legend()
    # plt.grid(True)
    # plt.show()


    df_sorted['Quantile_Position'] = ratio_to_quantile_func(df_sorted[ratio_column])
    df_sorted['Transformed'] = transformation_function(df_sorted['Quantile_Position'])


    transformed_df = df_sorted[['original_index', 'Transformed']]
    data = data.merge(transformed_df, on='original_index', how='left')
    new_column_name = f"transformed_{ratio_column}"
    data.rename(columns={'Transformed': new_column_name}, inplace=True)


    data.drop(columns=['original_index'], inplace=True)

    # Step 10: Save the transformation functions if requested
    if save_functions:
        transformation_func_filename = os.path.join(functions_dir, f"transformation_function_{ratio_column}.pkl")
        with open(transformation_func_filename, 'wb') as f:
            pickle.dump(transformation_function, f)

        ratio_to_quantile_func_filename = os.path.join(functions_dir, f"ratio_to_quantile_func_{ratio_column}.pkl")
        with open(ratio_to_quantile_func_filename, 'wb') as f:
            pickle.dump(ratio_to_quantile_func, f)
        print(f"Functions saved for ratio '{ratio_column}' in directory '{functions_dir}'")

    return data


# In[6]:


years = sorted(df['fs_year'].unique())

for i in range(len(years) - 2):
    train_years = years[:i+2] if i > 0 else years[:2]
    test_year = years[i+2]

    print(f"\n--- Iteration {i+1} ---")
    print(f"Training Years: {train_years}")
    print(f"Test Year: {test_year}")

    # Split the data
    train_data = df[df['fs_year'].isin(train_years)].copy()
    test_data = df[df['fs_year'] == test_year].copy()

    # Reset indices to ensure alignment
    y_train = train_data['default'].reset_index(drop=True)
    y_test = test_data['default'].reset_index(drop=True)

    variables = ["roa", "operating_profit_margin", "ocf_ratio",
                 "cash_ratio", "asset_coverage_ratio", "asset_turnover_ratio",
                 "liability_to_asset", "dscr", "tie"]

    buckets = {}
    buckets['roa'] = (21, 0.2)
    buckets['roe'] = (51, 0.1)
    buckets['operating_profit_margin'] = (31, 0.3)
    buckets['ocf_ratio'] = (21, 0.31)
    buckets['cash_ratio'] = (51, 0.2)
    buckets['asset_coverage_ratio'] = (17, 0.35)
    buckets['asset_turnover_ratio'] = (51, 0.2)
    buckets['liability_to_asset'] = (7, 0.5)
    buckets['dscr'] = (41, 0.2)
    buckets['tie'] = (41, 0.2)

    transformed_variables = []

    # Transform variables in train_data and save transformation functions
    for var in variables:
        print(f"Transforming {var}")
        train_data = transform_ratio(
            data=train_data,
            ratio_column=var,
            default_column='default',
            k=31,
            frac=0.3,
            save_functions=True,
            functions_dir='functions/'
        )
        transformed_var = f'transformed_{var}'
        if transformed_var not in transformed_variables:
            transformed_variables.append(transformed_var)

    # Prepare X_train
    X_train = train_data[transformed_variables].reset_index(drop=True)

    # Load transformation functions
    ratio_to_quantiles = {}
    transformation_functions = {}

    for var in variables:
        try:
            with open(f'functions/ratio_to_quantile_func_{var}.pkl', 'rb') as f:
                ratio_to_quantiles[var] = pickle.load(f)
            with open(f'functions/transformation_function_{var}.pkl', 'rb') as f:
                transformation_functions[var] = pickle.load(f)
        except FileNotFoundError as e:
            print(f"Transformation functions for '{var}' not found: {e}")
            continue

    # Apply transformations to test_data
    for var in variables:
        print(f"Applying transformation for {var} on test data")
        RTQ = ratio_to_quantiles.get(var)
        TF = transformation_functions.get(var)
        if RTQ is None or TF is None:
            print(f"Skipping {var} due to missing transformation functions.")
            continue
        # Handle missing values in test_data[var]
        if test_data[var].isnull().any():
            test_data[var].fillna(train_data[var].median(), inplace=True)
            print(f"Filled missing values in {var} with train data median.")
        # Map ratio values to quantile positions
        quantile_positions = RTQ(test_data[var].values)
        # Apply the transformation function
        transformed_values = TF(quantile_positions)
        test_data[f'transformed_{var}'] = transformed_values

    # Prepare X_test
    X_test = test_data[transformed_variables].reset_index(drop=True)

    # Add constant term to the predictors
    X_train_const = sm.add_constant(X_train)
    X_test_const = sm.add_constant(X_test)

    # Fit logistic regression model
    logit_model = sm.Logit(y_train, X_train_const)
    try:
        logit_result = logit_model.fit(disp=False)
        print("Logistic regression model fitted successfully.")
    except Exception as e:
        print(f"Logistic regression failed: {e}")
        continue

    # Predict probabilities on test data
    try:
        y_prob_logit = logit_result.predict(X_test_const)
    except Exception as e:
        print(f"Prediction failed: {e}")
        continue

    # Compute ROC AUC
    try:
        fpr_logit, tpr_logit, _ = roc_curve(y_test, y_prob_logit)
        roc_auc_logit = auc(fpr_logit, tpr_logit)
    except Exception as e:
        print(f"ROC AUC computation failed: {e}")
        continue

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr_logit, tpr_logit, color='darkorange', lw=2, label=f'Logit ROC curve (area = {roc_auc_logit:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC for Logistic Regression ({test_year})')
    plt.legend(loc='lower right')
    plt.show()

    print(f"Logistic Regression ROC AUC for {test_year}: {roc_auc_logit:.2f}")


# In[7]:


years = sorted(df['fs_year'].unique())

for i in range(len(years) - 2):
    train_years = years[:i+2] if i > 0 else years[:2]
    test_year = years[i+2]

    print(f"\n--- Iteration {i+1} ---")
    print(f"Training Years: {train_years}")
    print(f"Test Year: {test_year}")

    # Split the data
    train_data = df[df['fs_year'].isin(train_years)].copy()
    test_data = df[df['fs_year'] == test_year].copy()

    # Reset indices to ensure alignment
    y_train = train_data['default'].reset_index(drop=True)
    y_test = test_data['default'].reset_index(drop=True)

    variables = ["roa", "operating_profit_margin", "ocf_ratio",
                 "cash_ratio", "asset_coverage_ratio", "asset_turnover_ratio",
                 "liability_to_asset", "dscr", "tie"]

    transformed_variables = []

    for var in variables:
        print(f"Transforming {var}")
        train_data = transform_ratio(
            data=train_data,
            ratio_column=var,
            default_column='default',
            k=51,
            frac=0.2,
            save_functions=True,
            functions_dir='functions/'
        )
        transformed_var = f'transformed_{var}'
        if transformed_var not in transformed_variables:
            transformed_variables.append(transformed_var)


    X_train = train_data[transformed_variables].reset_index(drop=True)

    # Load transformation functions
    ratio_to_quantiles = {}
    transformation_functions = {}

    for var in variables:
        try:
            with open(f'functions/ratio_to_quantile_func_{var}.pkl', 'rb') as f:
                ratio_to_quantiles[var] = pickle.load(f)
            with open(f'functions/transformation_function_{var}.pkl', 'rb') as f:
                transformation_functions[var] = pickle.load(f)
        except FileNotFoundError as e:
            print(f"Transformation functions for '{var}' not found: {e}")
            continue

    # Apply transformations to test_data
    for var in variables:
        print(f"Applying transformation for {var} on test data")
        RTQ = ratio_to_quantiles.get(var)
        TF = transformation_functions.get(var)
        if RTQ is None or TF is None:
            print(f"Skipping {var} due to missing transformation functions.")
            continue
        # Handle missing values in test_data[var]
        if test_data[var].isnull().any():
            test_data[var].fillna(train_data[var].mean(), inplace=True)
        # Map ratio values to quantile positions
        quantile_positions = RTQ(test_data[var].values)
        # Apply the transformation function
        transformed_values = TF(quantile_positions)
        test_data[f'transformed_{var}'] = transformed_values
        if f'transformed_{var}' not in transformed_variables:
            transformed_variables.append(f'transformed_{var}')

    # Ensure that test_data contains all transformed variables
    missing_transformed_vars = set(transformed_variables) - set(test_data.columns)
    if missing_transformed_vars:
        print(f"Missing transformed variables in test data: {missing_transformed_vars}")
        for var in missing_transformed_vars:
            test_data[var] = 0
            print(f"Filled missing transformed variable '{var}' with 0.")


    X_test = test_data[transformed_variables].reset_index(drop=True)


    X_train_const = sm.add_constant(X_train)
    X_test_const = sm.add_constant(X_test)



    tree_model = DecisionTreeClassifier(max_depth=20, max_leaf_nodes=80, random_state=42)  # You can set other hyperparameters as needed
    try:
        tree_model.fit(X_train, y_train)
        print("Decision Tree model fitted successfully.")
    except Exception as e:
        print(f"Decision Tree fitting failed: {e}")
        continue


    try:
        y_prob_tree = tree_model.predict_proba(X_test)[:, 1]
    except Exception as e:
        print(f"Prediction failed: {e}")
        continue


    try:
        fpr_tree, tpr_tree, _ = roc_curve(y_test, y_prob_tree)
        roc_auc_tree = auc(fpr_tree, tpr_tree)
    except Exception as e:
        print(f"ROC AUC computation failed: {e}")
        continue

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr_tree, tpr_tree, color='green', lw=2, label=f'Decision Tree ROC curve (area = {roc_auc_tree:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC for Decision Tree ({test_year})')
    plt.legend(loc='lower right')
    plt.show()

    print(f"Decision Tree ROC AUC for {test_year}: {roc_auc_tree:.2f}")


# In[8]:


years = sorted(df['fs_year'].unique())

for i in range(len(years) - 2):
    train_years = years[:i+2] if i > 0 else years[:2]
    test_year = years[i+2]

    print(f"\n--- Iteration {i+1} ---")
    print(f"Training Years: {train_years}")
    print(f"Test Year: {test_year}")

    # Split the data
    train_data = df[df['fs_year'].isin(train_years)].copy()
    test_data = df[df['fs_year'] == test_year].copy()

    # Reset indices to ensure alignment
    y_train = train_data['default'].reset_index(drop=True)
    y_test = test_data['default'].reset_index(drop=True)

    variables = ["roa", "operating_profit_margin", "ocf_ratio",
                 "cash_ratio", "asset_coverage_ratio", "asset_turnover_ratio",
                 "liability_to_asset", "dscr", "tie"]

    transformed_variables = []

    # Transform variables in train_data and save transformation functions
    for var in variables:
        print(f"Transforming {var}")
        train_data = transform_ratio(
            data=train_data,
            ratio_column=var,
            default_column='default',
            k=51,  # Adjust k based on your data
            frac=0.2,
            save_functions=True,
            functions_dir='functions/'
        )
        transformed_var = f'transformed_{var}'
        if transformed_var not in transformed_variables:
            transformed_variables.append(transformed_var)

    # Prepare X_train
    X_train = train_data[transformed_variables].reset_index(drop=True)

    # Load transformation functions
    ratio_to_quantiles = {}
    transformation_functions = {}

    for var in variables:
        try:
            with open(f'functions/ratio_to_quantile_func_{var}.pkl', 'rb') as f:
                ratio_to_quantiles[var] = pickle.load(f)
            with open(f'functions/transformation_function_{var}.pkl', 'rb') as f:
                transformation_functions[var] = pickle.load(f)
        except FileNotFoundError as e:
            print(f"Transformation functions for '{var}' not found: {e}")
            continue

    # Apply transformations to test_data
    for var in variables:
        print(f"Applying transformation for {var} on test data")
        RTQ = ratio_to_quantiles.get(var)
        TF = transformation_functions.get(var)
        if RTQ is None or TF is None:
            print(f"Skipping {var} due to missing transformation functions.")
            continue
        # Handle missing values in test_data[var]
        if test_data[var].isnull().any():
            test_data[var].fillna(train_data[var].median(), inplace=True)
            print(f"Filled missing values in {var} with train data median.")
        # Map ratio values to quantile positions
        quantile_positions = RTQ(test_data[var].values)
        # Apply the transformation function
        transformed_values = TF(quantile_positions)
        test_data[f'transformed_{var}'] = transformed_values
    # Prepare X_test
    X_test = test_data[transformed_variables].reset_index(drop=True)

    # Add constant term to the predictors
    X_train_const = sm.add_constant(X_train)
    X_test_const = sm.add_constant(X_test)


    gam = LogisticGAM(s(0 , n_splines=20, lam=0.1) + s(1 , n_splines=20, lam=0.1) + s(2 , n_splines=20, lam=0.1) + s(3, n_splines=20, lam=0.1) + s(4, n_splines=20, lam=0.1) + s(5, n_splines=20, lam=0.1) + s(6, n_splines=20, lam=0.1) + s(7, n_splines=20, lam=0.1) + s(8, n_splines=20, lam=0.1))
    gam.fit(X_train_const[transformed_variables], y_train)

    y_prob = gam.predict_proba(X_test_const[transformed_variables])
    y_pred = gam.predict(X_test_const[transformed_variables])

    # Evaluate the model
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='green', lw=2, label=f'GAM ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC for GAM  ({test_year})')
    plt.legend(loc='lower right')
    plt.show()
    print(f"GAM ROC AUC for {test_year}: {roc_auc:.2f}")

    with open('gam.pkl', 'wb') as file:
        pickle.dump(gam, file)

