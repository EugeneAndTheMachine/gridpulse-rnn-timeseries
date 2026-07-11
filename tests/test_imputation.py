"""Tests cho imputation module."""
import numpy as np
import pandas as pd
import pytest
from gridpulse.imputation.simple_imputers import ForwardFillImputer, MeanImputer
from gridpulse.imputation.interpolation import LinearInterpolationImputer
from gridpulse.imputation.knn_imputer import KNNImputer
from gridpulse.imputation.evaluate_imputation import compute_imputation_metrics


@pytest.fixture
def sample_df_with_missing():
    """DataFrame với missing values ở vị trí biết trước."""
    np.random.seed(42)
    data = np.random.randn(100, 3)
    df = pd.DataFrame(data, columns=["A", "B", "C"])
    df.iloc[10:15, 0] = np.nan  # missing block trong cột A
    df.iloc[[20, 25, 30], 1] = np.nan  # missing random trong cột B
    return df


@pytest.mark.parametrize("ImputerClass", [
    ForwardFillImputer, MeanImputer, LinearInterpolationImputer,
])
def test_imputer_removes_all_nans(ImputerClass, sample_df_with_missing):
    imputer = ImputerClass()
    result = imputer.fit_transform(sample_df_with_missing)
    assert not result.isnull().any().any(), f"{ImputerClass.__name__} left NaN values"


def test_imputer_preserves_observed_values(sample_df_with_missing):
    """Impute không được thay đổi giá trị observed."""
    original = sample_df_with_missing.copy()
    result = LinearInterpolationImputer().fit_transform(sample_df_with_missing)

    # Ở positions không NaN, giá trị phải giữ nguyên
    observed_mask = ~original.isnull()
    for col in original.columns:
        col_mask = observed_mask[col]
        np.testing.assert_array_almost_equal(
            original.loc[col_mask, col].values,
            result.loc[col_mask, col].values,
        )


def test_compute_metrics_zero_error_on_perfect_imputation():
    y_true = np.array([[1.0, 2.0], [3.0, 4.0]])
    y_imputed = y_true.copy()
    mask = np.array([[True, False], [False, True]])

    metrics = compute_imputation_metrics(y_true, y_imputed, mask)
    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["n_masked"] == 2