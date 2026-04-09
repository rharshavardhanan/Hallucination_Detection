import pandas as pd
import numpy as np
import os
from config import OUTPUT_DIR


def generate_submission(lgbm_model, cat_model, X_test, test_df):
    lgbm_proba = lgbm_model.predict_proba(X_test)[:, 1]
    cat_proba = cat_model.predict_proba(X_test)[:, 1]
    ensemble_proba = (lgbm_proba + cat_proba) / 2

    submission = pd.DataFrame({
        'Id': test_df['Id'],
        'Target': ensemble_proba
    })
    path = os.path.join(OUTPUT_DIR, "submission.csv")
    submission.to_csv(path, index=False)
    print(f"Submission saved: {path}")
    print(submission.head())
    return submission
