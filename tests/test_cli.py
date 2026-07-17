from market_regime import cli
from market_regime.config import ExperimentConfig


def _unexpected_call(name: str):
    def fail(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError(f"{name} must not be called")

    return fail


def test_predict_live_cli_loads_a_saved_model_without_training(tmp_path, monkeypatch) -> None:
    config = ExperimentConfig()
    dataset = object()
    result = object()
    calls = []

    monkeypatch.setattr(cli, "load_config", lambda path: config)
    monkeypatch.setattr(cli, "_real_dataset", lambda loaded, refresh: dataset)
    monkeypatch.setattr(cli, "train_live_model", _unexpected_call("train_live_model"))
    monkeypatch.setattr(cli, "run_live_forecast", _unexpected_call("run_live_forecast"))

    def predict(received_dataset, received_config, model_dir, output_dir):  # noqa: ANN001
        calls.append((received_dataset, received_config, model_dir, output_dir))
        return result

    monkeypatch.setattr(cli, "predict_live_regime", predict)
    monkeypatch.setattr(cli, "_print_forecast", lambda received: calls.append(received))
    model_dir = tmp_path / "model"
    output_dir = tmp_path / "predictions"

    exit_code = cli.main(
        [
            "predict-live",
            "--config",
            "config.yaml",
            "--model-dir",
            str(model_dir),
            "--output",
            str(output_dir),
            "--refresh",
        ]
    )

    assert exit_code == 0
    assert calls == [(dataset, config, model_dir, output_dir), result]


def test_train_live_cli_saves_a_model_without_predicting(tmp_path, monkeypatch) -> None:
    config = ExperimentConfig()
    dataset = object()
    result = object()
    calls = []

    monkeypatch.setattr(cli, "load_config", lambda path: config)
    monkeypatch.setattr(cli, "_real_dataset", lambda loaded, refresh: dataset)
    monkeypatch.setattr(cli, "predict_live_regime", _unexpected_call("predict_live_regime"))
    monkeypatch.setattr(cli, "run_live_forecast", _unexpected_call("run_live_forecast"))

    def train(received_dataset, received_config, model_dir, *, verbose):  # noqa: ANN001
        calls.append((received_dataset, received_config, model_dir, verbose))
        return result

    monkeypatch.setattr(cli, "train_live_model", train)
    monkeypatch.setattr(cli, "_print_live_training", lambda received: calls.append(received))
    model_dir = tmp_path / "model"

    exit_code = cli.main(
        [
            "train-live",
            "--config",
            "config.yaml",
            "--model-dir",
            str(model_dir),
            "--verbose",
        ]
    )

    assert exit_code == 0
    assert calls == [(dataset, config, model_dir, True), result]


def test_forecast_cli_uses_the_combined_backward_compatible_workflow(tmp_path, monkeypatch) -> None:
    config = ExperimentConfig()
    dataset = object()
    result = object()
    calls = []

    monkeypatch.setattr(cli, "load_config", lambda path: config)
    monkeypatch.setattr(cli, "_real_dataset", lambda loaded, refresh: dataset)
    monkeypatch.setattr(cli, "train_live_model", _unexpected_call("train_live_model"))
    monkeypatch.setattr(cli, "predict_live_regime", _unexpected_call("predict_live_regime"))

    def forecast(received_dataset, received_config, output_dir, *, verbose):  # noqa: ANN001
        calls.append((received_dataset, received_config, output_dir, verbose))
        return result

    monkeypatch.setattr(cli, "run_live_forecast", forecast)
    monkeypatch.setattr(cli, "_print_forecast", lambda received: calls.append(received))
    output_dir = tmp_path / "forecast"

    exit_code = cli.main(
        [
            "forecast",
            "--config",
            "config.yaml",
            "--output",
            str(output_dir),
            "--verbose",
        ]
    )

    assert exit_code == 0
    assert calls == [(dataset, config, output_dir, True), result]
