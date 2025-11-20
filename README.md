# Credit Card Fraud Detection with XGBoost

## Project Overview
This project builds a machine learning model to detect fraudulent credit card transactions. It handles extreme class imbalance using **XGBoost** with a weighted loss function (`scale_pos_weight`).

## Key Results
* **Algorithm:** XGBoost
* **Technique:** Class Weighting (`scale_pos_weight`)
* **Business Value:** Reduced expected operational costs by **27%** compared to the Focal Loss approach.
* **Performance:** Recall: ~84% | Precision: ~94%

## How to run

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
2. **Download Data:**
Download creditcard.csv from Kaggle and place it in the root folder.  
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data
4. **Run the code.**
  
  
