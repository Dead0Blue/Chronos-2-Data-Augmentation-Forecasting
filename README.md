# Chronos-2 Data Augmentation & Forecasting

This repository implements the data augmentation framework for telecom traffic using an Interrupted Poisson Process (IPP), as described in the KTH paper: *Digital Twin Assisted Risk-Aware Sleep Mode Management Using Deep Q-Networks*.

## Overview

The project focuses on:
- **Data Augmentation**: Converting 5-minute averaged traffic from the Bouygues dataset into 30-second instantaneous traffic using an IPP model.
- **Forecasting**: Comparing the state-of-the-art **Chronos-2** (T5-small version) with a traditional **SARIMA** baseline.

## Results

Evaluation across all sectors with stochastic augmentation yields the following average performance:
- **Chronos-2 RMSE**: ~11.50
- **SARIMA RMSE**: ~11.25

*Note: Results may vary slightly due to the random nature of the data augmentation process.*

## Project Structure

- `augmentation.py`: Implements the IPP simulation logic.
- `forecast_eval.py`: Handles model loading, prediction, and performance comparison across all sectors.
- `project/`: Contains the datasets (historical and augmented) and visualization plots.

## Setup

1. Install dependencies:
   ```bash
   pip install chronos-forecasting statsmodels scikit-learn matplotlib torch pandas numpy
   ```
2. Run data augmentation:
   ```bash
   python augmentation.py
   ```
3. Run evaluation:
   ```bash
   python forecast_eval.py
   ```

## References

- [KTH Paper PDF](KTH.pdf)
- Section II-C and Figure 6(a) for the IPP framework.
