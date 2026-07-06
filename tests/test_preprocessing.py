# tests/test_preprocessing.py

def test_calendar_features_cyclical_range():
    """Sin/cos values must be in [-1, 1]."""
    from gridpulse.features.calendar_features import add_calendar_features
    import pandas as pd

    df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=100, freq="h")})
    df = add_calendar_features(df)

    assert df["hour_sin"].between(-1, 1).all()
    assert df["hour_cos"].between(-1, 1).all()


def test_forecasting_dataset_length():
    """Dataset length should match expected window count."""
    import numpy as np
    from gridpulse.datasets.forecasting_dataset import ForecastingDataset

    data = np.random.randn(1000, 7)
    ds = ForecastingDataset(data, input_len=96, forecast_horizon=24, stride=1)
    assert len(ds) == 1000 - 96 - 24 + 1