from module_vae.tracking import flatten


def test_flatten_nested_metrics_and_drops_non_numbers():
    metrics = {"auc": 0.75, "ranking": {"precision_at_10": 0.4}, "name": "x", "ok": True}

    assert flatten(metrics, "val") == {"val_auc": 0.75, "val_ranking_precision_at_10": 0.4}
