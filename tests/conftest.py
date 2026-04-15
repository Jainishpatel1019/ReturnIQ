import pytest
import pandas as pd
import numpy as np
import pathlib

@pytest.fixture(scope="session", autouse=True)
def ensure_test_fixtures():
    """Generates synthetic data for tests if the real pipeline output is missing."""
    data_dir = pathlib.Path("data/processed")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # fixture 1: final_dashboard_data.parquet
    final_path = data_dir / "final_dashboard_data.parquet"
    if not final_path.exists():
        print("Generating synthetic final_dashboard_data.parquet for CI tests...")
        df = pd.DataFrame({
            'seller_id': [f'S{i:03d}' for i in range(50)],
            'category': np.random.choice(['Electronics', 'Home_and_Kitchen', 'Clothing_Shoes_and_Jewelry'], 50),
            'proxy_return_rate': np.random.uniform(0.01, 0.4, 50),
            'cate': np.random.uniform(-0.05, 0.1, 50),
            'cate_lo': np.random.uniform(-0.1, 0.05, 50),
            'cate_hi': np.random.uniform(0.05, 0.2, 50),
            'ols_pred': np.random.uniform(0.01, 0.4, 50),
            'total_reviews': np.random.randint(10, 1000, 50),
            'rating_variance': np.random.uniform(0, 1, 50),
            'seller_quality_score': np.random.uniform(0, 1, 50),
            'narrative': ["This is a synthetic narrative for testing purposes. It should be long enough to pass."]*50
        })
        for cl in ['Clothing_Shoes_and_Jewelry', 'Electronics', 'Home_and_Kitchen']:
            df[f'shap_cat_{cl}'] = np.random.normal(0, 0.01, 50)
        df.to_parquet(final_path, index=False)

    # fixture 2: ols_predictions.parquet
    ols_path = data_dir / "ols_predictions.parquet"
    if not ols_path.exists():
        df_ols = pd.DataFrame({
            'seller_id': [f'S{i:03d}' for i in range(50)],
            'ols_pred': np.random.uniform(0.01, 0.4, 50)
        })
        df_ols.to_parquet(ols_path, index=False)

    # fixture 3: cate_results.parquet
    cate_path = data_dir / "cate_results.parquet"
    if not cate_path.exists():
         df_cate = pd.read_parquet(final_path)
         df_cate.to_parquet(cate_path, index=False)

    # fixture 4: returns.duckdb
    db_path = data_dir / "returns.duckdb"
    if not db_path.exists():
        import duckdb
        con = duckdb.connect(str(db_path))
        con.execute("CREATE TABLE reviews (id INTEGER)")
        con.execute("CREATE TABLE products (id INTEGER)")
        con.execute("CREATE TABLE sellers (id INTEGER)")
        con.execute("CREATE TABLE cohort_stats (seller_id VARCHAR, category VARCHAR, total_reviews INTEGER, rating_variance FLOAT, proxy_return_rate FLOAT)")
        con.close()

    # fixture 5: seller_features_trimmed.parquet
    seller_path = data_dir / "seller_features_trimmed.parquet"
    if not seller_path.exists():
         df_seller = pd.read_parquet(final_path)
         df_seller.to_parquet(seller_path, index=False)
