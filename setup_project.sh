#!/bin/bash

# Create main project folder
mkdir -p battery_RUL

# Create subfolders
mkdir -p battery_RUL/data
mkdir -p battery_RUL/notebooks
mkdir -p battery_RUL/results/plots
mkdir -p battery_RUL/results/models
mkdir -p battery_RUL/results/forecasts

# Create empty Jupyter notebooks
touch battery_RUL/notebooks/01_EDA_Discharge.ipynb
touch battery_RUL/notebooks/02_Stationarity_Check_Discharge.ipynb
touch battery_RUL/notebooks/03_ARIMA_Model_Discharge.ipynb
touch battery_RUL/notebooks/04_Forecasting_Discharge.ipynb

# Create README.md and .gitignore
touch battery_RUL/README.md
touch battery_RUL/.gitignore

