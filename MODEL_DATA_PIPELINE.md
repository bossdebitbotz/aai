# LOB Forecasting Model: Data Pipeline and Input Guide

## 1. Overview

This document provides a comprehensive guide to the data pipeline for the attention-based limit order book (LOB) forecasting model. Its purpose is to ensure that data is processed consistently for both training and live inference.

The pipeline's primary goal is to convert raw, high-frequency LOB data from multiple exchanges and trading pairs into a unified, pre-processed format suitable for the spatiotemporal attention model. Adhering to this pipeline is **critical** for the model to function correctly, especially when feeding it live data from a trading bot.

## 2. File & Directory Structure

-   `multi_exchange_lob_collector.py`:
    -   Collects LOB data from exchanges.
    -   Saves raw hourly data to `data/raw/`.
    -   Performs initial resampling to 5-second intervals, saving to `data/processed/`.

-   `prepare_model_data.py`:
    -   Executes the main data preparation logic described below.
    -   Reads from `data/processed/`.
    -   Outputs the final, model-ready datasets and processing artifacts to `data/final/`.

-   `/data`:
    -   `raw/`: Raw, partitioned Parquet files from the collector.
    -   `processed/`: 5-second resampled Parquet files.
    -   `final/`: The directory containing all the necessary inputs for model training and inference.

## 3. Phase 1: Training Data Preparation (`prepare_model_data.py`)

This phase converts the 5-second resampled data into model-ready training, validation, and test sets.

**Step 1: Load All Resampled Data**
-   The script begins by finding all `_resampled_5s.parquet` files within the `data/processed/` directory for the configured exchanges and trading pairs.

**Step 2: Create a Unified Multivariate Time Series**
-   Data for each instrument (e.g., `binance_spot_BTC-USDT`) is loaded.
-   Column names are made unique by prefixing them (e.g., `bid_price_1` becomes `binance_spot_BTC-USDT_bid_price_1`).
-   All individual DataFrames are concatenated column-wise into a single, wide DataFrame.
-   This DataFrame is re-indexed to a perfect 5-second frequency (`freq='5S'`) to ensure consistency. Any missing timestamps are forward-filled (`ffill`).

**Step 3: Stationary Transformation (Prices Only)**
-   To stabilize the price data, a **percent-change** transformation is applied to every price column.
-   Formula: `p_t_percent = (p_t - p_{t-1}) / p_{t-1}`. The first row's NaN values are filled with 0.

**Step 4: Min-Max Scaling**
-   **Every single column** (percent-changed prices and raw volumes) is scaled to a `[0, 1]` range.
-   A separate `MinMaxScaler` object is `fit_transform`'ed for each column.
-   **Crucially, these fitted scalers are saved as a dictionary to `data/final/scalers.gz`.** This is essential for reversing the process and for processing live data.

**Step 5: Sequence Generation**
-   The scaled DataFrame is converted into overlapping sequences for the model.
-   **Context (Input) Length**: `120` time steps (10 minutes of data).
-   **Target (Output) Length**: `24` time steps (2 minutes of data).
-   The final array shape is `(num_sequences, sequence_length, num_features)`.

**Step 6: Data Splitting and Saving**
-   The generated sequences are split chronologically into training (60%), validation (20%), and testing (20%) sets.
-   These sets are saved as compressed NumPy archives (`.npz`) in the `data/final/` directory.

## 4. Key Artifacts in `data/final/`

The `prepare_model_data.py` script generates the following critical files in `data/final/`. These are the **only** files needed for training and inference.

-   `train.npz`, `validation.npz`, `test.npz`:
    -   **Description**: Compressed files containing the data splits. Each contains two arrays: `x` (the context/input sequences) and `y` (the target/output sequences).

-   `scalers.gz`:
    -   **Description**: A compressed dictionary where keys are the full column names (e.g., `binance_spot_BTC-USDT_bid_price_1`) and values are the corresponding fitted `sklearn.preprocessing.MinMaxScaler` objects.
    -   **Usage**: **Essential for inference.** Used to scale live data and to inverse-transform the model's predictions back to a human-readable scale.

-   `columns.gz`:
    -   **Description**: A compressed Python list containing the names of all 240 columns in their exact order as they appear in the feature dimension of the NumPy arrays.
    -   **Usage**: **Essential for inference.** Ensures that live data passed to the model has the exact same feature order as the training data.

## 5. Phase 2: Inference Pipeline for Live Data (Trading Bot Guide)

To use the trained model, the trading bot must **perfectly replicate** the data preparation steps using the saved artifacts.

**Input Requirement**: The bot must have access to the last **10 minutes** of 5-second LOB data for **all 12 instruments** that the model was trained on.

**Step-by-Step Inference Process:**

1.  **Load Artifacts**:
    -   Load the column order: `column_order = joblib.load('data/final/columns.gz')`
    -   Load the scalers: `scalers = joblib.load('data/final/scalers.gz')`

2.  **Prepare Live DataFrame**:
    -   Assemble the 120 most recent 5-second data points into a single pandas DataFrame.
    -   Ensure the DataFrame has all 240 columns required by the model.
    -   **CRITICAL**: Re-order the DataFrame columns to match the `column_order` list: `live_df = live_df[column_order]`.

3.  **Apply Transformations**:
    -   **a. Percent-Change Prices**: Identify the price columns and apply the percent-change transformation.
    -   **b. Scale All Features**: Loop through each column in `column_order`. For each column, use its corresponding scaler from the `scalers` dictionary to transform the data. **Use `scaler.transform()`, NOT `fit_transform()`.**

4.  **Create Model Input**:
    -   Convert the processed DataFrame to a NumPy array.
    -   Reshape the array to `(1, 120, 240)` to create a single batch for the model.

5.  **Get Prediction**:
    -   Feed the `(1, 120, 240)` tensor into the model.
    -   The model will output a prediction tensor of shape `(1, 24, 240)`.

6.  **Inverse Transform Prediction**:
    -   The output prediction is in the scaled `[0, 1]` range. To make it useful, it must be inverse-transformed.
    -   Reshape the prediction to `(24, 240)`.
    -   For each of the 240 features, use the corresponding scaler's `.inverse_transform()` method to convert the predicted values back to their original scale (raw volumes and percent-changed prices).
    -   For the price columns, an additional step is required to reverse the percent-change calculation to get the actual predicted price levels. 