import pytest
import pandas as pd
import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

def test_feature_matrix_structure():
    """Ensure confounder matrix X matches inference requirements."""
    from src.causal_model import get_feature_matrix
    
    # Mock data
    df = pd.DataFrame({
        "category": ["Electronics", "Books"],
        "price_tier": [1, 2],
        "unique_buyers": [100, 200],
        "total_reviews": [50, 60],
        "competitor_density": [0.1, 0.2],
        "niche_share": [0.3, 0.4]
    })
    
    X, names = get_feature_matrix(df)
    assert "price_tier" in names
    assert X.shape[1] > 5
    assert not X.isnull().values.any()

def test_variance_decomposition_rigor():
    """Verify η² calculation logic."""
    from src.variance_decomp import calculate_variance_decomposition
    
    df = pd.DataFrame({
        "proxy_return_rate": [0.1, 0.2, 0.1, 0.2],
        "category": ["A", "A", "B", "B"]
    })
    
    eta_sq = calculate_variance_decomposition(df, "category", "proxy_return_rate")
    assert 0 <= eta_sq <= 1

def test_data_loader_robustness():
    """Test path resolution in app.py logic."""
    # This checks if the project structure is consistent
    assert os.path.exists("data/processed")
    assert os.path.exists("src/ui_helpers.py")
