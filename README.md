# probability-of-default-predictor

A machine learning project to predict the **Probability of Default (PD)** for firms applying for loans. This tool enables banks to make data-driven decisions on loan approvals and interest rate calculations by leveraging financial ratios and advanced predictive models.

## Overview

This project is designed to enhance the underwriting process at **Banca Massiccia**, a large Italian bank, by predicting the one-year probability of default for prospective borrowers. The model combines financial domain expertise with machine learning techniques to ensure accurate, interpretable, and actionable predictions.

## Features

- **Predict Default Probability**: Uses a Generalized Additive Model (GAM) for smooth, interpretable predictions and decision trees for non-linear insights.
- **Explainable Results**: Combines domain knowledge and machine learning to offer interpretable outcomes aligned with financial principles.
- **Robust Pipeline**: Implements non-parametric transformations and Bayesian adjustments for reliable, calibrated predictions.

## Key Financial Ratios

The model leverages various financial ratios as predictors:
- **Profitability Ratios**: ROA (Return on Assets), Operating Profit Margin
- **Liquidity Ratios**: Cash Ratio, Operating Cash Flow Ratio
- **Coverage Ratios**: Asset Coverage Ratio
- **Efficiency Ratios**: Asset Turnover Ratio
- **Leverage Ratios**: Liability-to-Asset Ratio, DSCR (Debt Service Coverage Ratio), TIE (Times Interest Earned)

## Technologies Used

- **Programming Language**: Python
- **Libraries**: 
  - Data Processing: Pandas, NumPy
  - Machine Learning: Scikit-learn, PyGAM, Statsmodels
  - Visualization: Matplotlib, Seaborn
- **Techniques**:
  - **Non-parametric transformations for feature engineering:** Non-parametric transformations enhance feature representation by capturing complex, non-linear relationships between input features and the target variable. Inspired by techniques from **RiskCalc™**, this method involves sorting feature values into quantile buckets, calculating default rates for each quantile, and fitting a smooth, non-linear curve (LOESS smoothing) to map quantiles to their default rates. These transformations preserve interpretability while ensuring the model effectively learns meaningful patterns, improving its predictive power and alignment with financial intuition.
  - **Walk-forward testing harness for model evaluation:** Walk-forward testing is a robust method to evaluate model performance in real-world scenarios. In this approach, the data is split into sequential training and testing sets based on time, mimicking the process of making predictions on future data using past observations. For each iteration, the training set expands to include more recent data, while the testing set includes the next time period. This technique ensures the model is tested on unseen data that simulates actual future predictions, providing a reliable measure of its accuracy and robustness over time.
  - **Bayesian calibration for probability adjustment:** Bayesian calibration adjusts predicted probabilities to account for differences between the training data distribution and the real-world population. This technique refines raw model outputs by incorporating observed base rates of defaults, resulting in better-aligned probabilities with the true default rates. By applying Bayesian adjustment, the calibrated probabilities become more reliable for decision-making, especially in dynamic financial environments where data distribution may drift over time.


## Structure

### 1. Preprocessing
- Handles missing values with financial domain logic.
- Generates key financial ratios like ROA, DSCR, and Cash Ratio.
- Applies transformations to ensure interpretability and reduce noise.

### 2. Modeling
- Implements Logistic GAM for smooth non-linear predictions.
- Decision Trees capture hierarchical contributions of variables.
- Walk-forward analysis for robust evaluation across different time periods.

### 3. Prediction
- Processes new data with pre-trained transformation functions.
- Calibrates predictions using Bayesian adjustment for population drift.

### 4. Deployment
- The pipeline integrates seamlessly with existing banking systems.
- CLI interface provided via `harness.py`.

## How to Use

1. **Preprocess Data**: Run the `preprocessor.py` script to clean and preprocess financial data.
2. **Train the Model**: Use `estimator.py` to fit and evaluate the model.
3. **Make Predictions**: Use `harness.py` with the following command:
   ```bash
   python harness.py --input_csv <path_to_input_csv> --output_csv <path_to_output_csv>
