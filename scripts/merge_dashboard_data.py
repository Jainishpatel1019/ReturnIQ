import pandas as pd
import os

MAIN_DATA = "data/processed/final_dashboard_data.parquet"
OLS_PRED  = "data/processed/ols_predictions.parquet"

def merge_data():
    if not os.path.exists(MAIN_DATA) or not os.path.exists(OLS_PRED):
        print("Missing files. Run OLS/Pipeline first.")
        return

    df_main = pd.read_parquet(MAIN_DATA)
    df_ols  = pd.read_parquet(OLS_PRED)

    print(f"Main size: {len(df_main)}, OLS size: {len(df_ols)}")
    
    # Merge ols_pred column
    if 'ols_pred' in df_main.columns:
        df_main = df_main.drop(columns=['ols_pred'])
    
    df_merged = df_main.merge(df_ols[['seller_id', 'ols_pred']], on='seller_id', how='left')
    
    # Fill remaining NaNs if any (though they should match)
    df_merged['ols_pred'] = df_merged['ols_pred'].fillna(df_merged['proxy_return_rate'] * 0.9) # Fallback

    df_merged.to_parquet(MAIN_DATA, index=False)
    print(f"✓ Successfully merged ols_pred into {MAIN_DATA}")
    print(f"Columns: {df_merged.columns.tolist()}")

if __name__ == "__main__":
    merge_data()
