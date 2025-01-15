# Sales Forecasting with Spark and Prophet

## Overview

This project aims to forecast sales using Facebook's Prophet for time series forecasting. The dataset used includes historical sales data from various stores and items across different states in Ecuador.

## Dataset

The dataset includes:

- `train.csv`: Historical sales records including store number, item number, date, and unit sales.
- `items.csv`: Details about items including item number, family, class, and perishable indicator.
- `stores.csv`: Store information including store number, city, state, type, and cluster.

## Project Structure

- Data Cleaning and Preparation: Utilizes PySpark for reading, cleaning, and merging datasets (train.csv, items.csv, stores.csv).
- Feature Engineering: Extracts relevant features such as item families, perishable items, store locations (city, state), etc.
- Prophet Model Tuning: Implements hyperparameter tuning for Prophet models using a grid search approach to optimize forecasting accuracy.
- Forecasting: Generates sales forecasts for different states and types of items (perishable vs non-perishable) using Prophet.
- Visualization: Plots actual vs. predicted sales trends for visualization and evaluation purposes.
