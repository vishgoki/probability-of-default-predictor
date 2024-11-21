import numpy as np
import pandas as pd


def preprocessor(data, isTraining=False):
    #### Preprocess individual variables with the financial knowledge

    # Date
    data['stmt_date'] = pd.to_datetime(data['stmt_date'], format='%Y-%m-%d')
    data['def_date'] = pd.to_datetime(data['def_date'], format='%d/%m/%Y')

    def assign_default(row):
        date_diff = (row['def_date'] - row['stmt_date']).days
        return 1 if 150 <= date_diff <= 515 else 0

    if isTraining:
        data['default'] = data.apply(assign_default, axis=1)

    data['prof_operations'] = np.where(
        data['prof_operations'].isnull() & data['rev_operating'].notnull() & data['COGS'].notnull(),
        data['rev_operating'] - data['COGS'],
        data['prof_operations'])

    data['prof_operations'] = np.where(
        data['prof_operations'].isnull() & data['roa'].notnull() & data['asst_tot'].notnull(),
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
        data['asst_tot'].isnull() & data['debt_st'].notnull() & data['debt_lt'].notnull() & data['liab_lt'].notnull() &
        data['liab_lt_emp'].notnull() & data['eqty_tot'].notnull(),
        data['debt_st'] + data['debt_lt'] + data['liab_lt'] + data['liab_lt_emp'] + data['eqty_tot'],
        data['asst_tot'])

    data['asst_tot'] = np.where(
        data['asst_tot'].isnull() & data['asst_current'].notnull() & data['asst_intang_fixed'].notnull() & data[
            'asst_tang_fixed'].notnull() & data['asst_fixed_fin'].notnull(),
        data['asst_current'] + data['asst_intang_fixed'] + data['asst_tang_fixed'] + data['asst_fixed_fin'],
        data['asst_tot'])

    data['debt_st'] = np.where(
        data['debt_st'].isnull() & data['asst_current'].notnull() & data['wc_net'].notnull(),
        data['asst_current'] - data['wc_net'],
        data['debt_st'])

    data['debt_st'] = np.where(
        data['debt_st'].isnull() & data['asst_tot'].notnull() & data['debt_lt'].notnull() & data['eqty_tot'].notnull() &
        data['liab_lt'].notnull() & data['liab_lt_emp'].notnull(),
        data['asst_tot'] - (data['debt_lt'] + data['eqty_tot'] + data['liab_lt'] + data['liab_lt_emp']),
        data['debt_st']
    )

    data['debt_lt'] = np.where(
        data['debt_lt'].isnull() & data['asst_tot'].notnull() & data['debt_st'].notnull() & data['eqty_tot'].notnull() &
        data['liab_lt'].notnull() & data['liab_lt_emp'].notnull(),
        data['asst_tot'] - (data['debt_st'] + data['eqty_tot'] + data['liab_lt'] + data['liab_lt_emp']),
        data['debt_lt']
    )

    data['eqty_tot'] = np.where(
        data['eqty_tot'].isnull() & data['asst_tot'].notnull() & data['debt_st'].notnull() & data['debt_lt'].notnull() &
        data['liab_lt'].notnull() & data['liab_lt_emp'].notnull(),
        data['asst_tot'] - (data['debt_st'] + data['debt_lt'] + data['liab_lt'] + data['liab_lt_emp']),
        data['eqty_tot']
    )

    data['liab_lt'] = np.where(
        data['liab_lt'].isnull() & data['asst_tot'].notnull() & data['debt_st'].notnull() & data['debt_lt'].notnull() &
        data['eqty_tot'].notnull() & data['liab_lt_emp'].notnull(),
        data['asst_tot'] - (data['debt_st'] + data['debt_lt'] + data['eqty_tot'] + data['liab_lt_emp']),
        data['liab_lt']
    )

    data['liab_lt_emp'] = np.where(
        data['liab_lt_emp'].isnull() & data['asst_tot'].notnull() & data['debt_st'].notnull() & data[
            'debt_lt'].notnull() & data['eqty_tot'].notnull() & data['liab_lt'].notnull(),
        data['asst_tot'] - (data['debt_st'] + data['debt_lt'] + data['eqty_tot'] + data['liab_lt']),
        data['liab_lt_emp']
    )

    #### RATIOS (Feature Engineering)

    ### Profitability Ratios

    # ROA
    # data['roa']

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
    data['asset_coverage_ratio'] = np.where(data['asset_coverage_ratio'] == -np.inf, -10600000,
                                            data['asset_coverage_ratio'])

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
    # data = data.replace([np.inf, -np.inf], 0)
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

    quantile_dict = {'roa': (-23.28, 30.59),
                     'roe': (-544.9262880748006, 249.11143822849405),
                     'operating_profit_margin': (-14.919108722722491, 0.8502205750286215),
                     'ocf_ratio': (-1.9166645487953502, 7.25539558239099),
                     'cash_ratio': (0.0, 17.406414301085707),
                     'asset_coverage_ratio': (0.5813068756524753, 67.19924399392627),
                     'asset_turnover_ratio': (0.0, 3.727608234014413),
                     'liability_to_asset': (0.0144803694337374, 1.1142658944767951),
                     'dscr': (-2.4387464594086694, 4.224639904211384),
                     'tie': (-65.14431106015142, 39.89361849290187)}

    if isTraining:
        variables = variables + ['default', 'fs_year']

    data = data[variables]
    for col in data.columns:
        if col in median_dict.keys():
            data = data.copy()
            data.loc[:, col] = data[col].clip(lower=quantile_dict.get(col)[0], upper=quantile_dict.get(col)[1])
            data.loc[:, col] = data[col].fillna(median_dict[col])

    print("Null values filled with medians")
    return data[variables]
