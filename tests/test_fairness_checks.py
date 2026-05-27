from tsf_paperkit.checks.fairness import check_config
from tsf_paperkit.runner.experiment import load_config, run_experiment


def test_train_only_scaler_and_warnings():
    cfg = load_config("configs/example.yaml")
    run_experiment(cfg)
    report = check_config(cfg)
    names = {c["name"]: c for c in report["checks"]}
    assert names["scaler_train_only"]["status"] == "pass"
    assert names["test_drop_last"]["status"] == "pass"
    assert names["results_required_columns"]["status"] == "pass"
    assert names["results_num_params_recorded"]["status"] == "pass"
    assert names["results_timing_recorded"]["status"] == "pass"
