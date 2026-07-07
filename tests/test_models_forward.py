"""Test that all models produce correct output shapes."""
import torch
import pytest
from gridpulse.models.rnn import RNNForecaster
from gridpulse.models.lstm import LSTMForecaster
from gridpulse.models.gru import GRUForecaster
from gridpulse.models.seq2seq import Seq2SeqForecaster


@pytest.fixture
def sample_batch():
    batch_size, input_len, num_features = 8, 96, 30
    return torch.randn(batch_size, input_len, num_features)


@pytest.mark.parametrize("ModelClass", [
    RNNForecaster, LSTMForecaster, GRUForecaster,
])
def test_encoder_decoder_output_shape(ModelClass, sample_batch):
    model = ModelClass(num_features=30, hidden_size=64, num_layers=1, forecast_horizon=24)
    model.eval()
    with torch.no_grad():
        output = model(sample_batch)
    assert output.shape == (8, 24)


def test_seq2seq_output_shape(sample_batch):
    model = Seq2SeqForecaster(num_features=30, hidden_size=64, num_layers=1, forecast_horizon=24)
    model.eval()
    with torch.no_grad():
        output = model(sample_batch)
    assert output.shape == (8, 24)


def test_model_save_load(tmp_path, sample_batch):
    model = LSTMForecaster(num_features=30, hidden_size=64, num_layers=1, forecast_horizon=24)
    model.eval()
    with torch.no_grad():
        out_before = model(sample_batch)

    # Save
    model.save_checkpoint(tmp_path / "test.pt")

    # Load into new model
    model2 = LSTMForecaster(num_features=30, hidden_size=64, num_layers=1, forecast_horizon=24)
    model2.load_checkpoint(tmp_path / "test.pt", torch.device("cpu"))
    model2.eval()
    with torch.no_grad():
        out_after = model2(sample_batch)

    torch.testing.assert_close(out_before, out_after)