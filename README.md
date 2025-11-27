# Credit Card Fraud Detection with XGBoost

## Project Overview
This project builds a machine learning model to detect fraudulent credit card transactions. It handles extreme class imbalance using **XGBoost** with a weighted loss function (`scale_pos_weight`).

## Key Results
* **Algorithm:** XGBoost
* **Technique:** Class Weighting (`scale_pos_weight`)
* **Business Value:** Optimized **Expected Operational Cost to 25,302 PLN** on the test sample.
* **Performance:** Recall: ~85% | Precision: ~91% | F1 Score: ~0.88%

## How to run

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
2. **Download Data:**  
Download creditcard.csv from Kaggle and place it in the root folder.  
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data
3. **Run the code.**
  
