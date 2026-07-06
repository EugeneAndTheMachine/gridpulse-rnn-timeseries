"""Test windowing logic — critical correctness check."""
import numpy as np
from gridpulse.preprocessing.windowing import create_windows


def test_window_shapes():
    """Output shapes must match expectations."""
    T, F = 1000, 7
    data = np.random.randn(T, F)
    input_len, horizon = 96, 24

    X, y = create_windows(data, input_len, horizon)

    expected_n = T - input_len - horizon + 1
    assert X.shape == (expected_n, input_len, F)
    assert y.shape == (expected_n, horizon)


def test_no_data_leakage():
    """Target values must NOT appear in input window."""
    T, F = 200, 3
    # Create data where each row is unique and identifiable
    data = np.arange(T * F).reshape(T, F).astype(float)
    input_len, horizon = 10, 5

    X, y = create_windows(data, input_len, horizon, target_col_idx=-1)

    # For first window: input is rows 0-9, target is rows 10-14 (last col)
    np.testing.assert_array_equal(X[0], data[0:10])
    np.testing.assert_array_equal(y[0], data[10:15, -1])

    # Input and target time ranges must NOT overlap
    assert X[0][-1, -1] != y[0][0]  # last input step ≠ first target step


def test_synthetic_mcar_rate():
    """MCAR injection should match requested rate (approximately)."""
    import pandas as pd
    from gridpulse.preprocessing.synthetic_missing import inject_mcar

    df = pd.DataFrame({"A": np.random.randn(10000), "B": np.random.randn(10000)})
    corrupted, mask = inject_mcar(df, columns=["A", "B"], missing_rate=0.1)

    actual_rate = mask.mean().mean()
    assert 0.08 < actual_rate < 0.12  # allow some variance