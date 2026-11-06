import numpy as np
import pytest

from project.mindmetrics.inference import FEATURES, load_artifacts, predict


def test_checkpoint_predictions_are_finite_and_repeatable():
    artifacts = load_artifacts()
    row = [(f["min"] + f["max"]) / 2 for f in FEATURES]
    np.testing.assert_array_equal(predict(row, artifacts), predict(row, artifacts))
    assert predict([row, row], artifacts).shape == (2,)


@pytest.mark.parametrize("values", [[1, 2], [float("nan")] * 15, [float("inf")] * 15])
def test_invalid_input_is_rejected(values):
    with pytest.raises(ValueError, match="finite values"):
        predict(values)


def test_artifacts_are_independent_of_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert np.isfinite(predict([f["min"] for f in FEATURES])).all()


def test_corrupt_scaler_is_rejected(tmp_path):
    np.savez(tmp_path / "scaler_params.npz", mean=np.zeros(15), scale=np.zeros(15))
    with pytest.raises(ValueError, match="positive scales"):
        load_artifacts(tmp_path)
