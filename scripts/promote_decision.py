"""
Record the promotion decision from REPORT.md in the registry.

The automated gate cannot make this call on its own: every sweep candidate passes,
and the AUC differences are inside the noise. The decision (see REPORT.md, section 3)
is to serve the fairest candidate, hgb-07, and keep logistic regression as the
challenger. Both go through promote_model first, so they are registered, gated and
tagged exactly like an automated promotion; the final alias move is then made by
hand and tagged with the reason, so the registry shows who decided what and why.

Usage:
    MLFLOW_TRACKING_URI=http://localhost:5000 python scripts/promote_decision.py
"""

from mlflow.tracking import MlflowClient

from pipeline.config import (
    CHALLENGER_ALIAS,
    CHAMPION_ALIAS,
    MLFLOW_EXPERIMENT_NAME,
    REGISTERED_MODEL_NAME,
)
from pipeline.registry import list_registered_models, promote_model
from pipeline.training import setup_mlflow


def latest_run(client: MlflowClient, experiment_id: str, run_name: str):
    runs = client.search_runs(
        [experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}' and metrics.roc_auc > 0",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise SystemExit(f"no finished run named {run_name}")
    return runs[0]


def main() -> None:
    setup_mlflow()
    client = MlflowClient()
    exp = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)

    decided = {}
    for name in ("logreg-02", "hgb-07"):
        run = latest_run(client, exp.experiment_id, name)
        result = promote_model(run.info.run_id, dict(run.data.metrics))
        decided[name] = result["version"]
        print(f"{name}: v{result['version']} -> {result['outcome']} (automated gate)")

    champion, challenger = decided["hgb-07"], decided["logreg-02"]
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS, champion)
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, CHALLENGER_ALIAS, challenger)
    client.set_model_version_tag(
        REGISTERED_MODEL_NAME, champion, "promotion_reason",
        "manual review: AUC within noise of best (-0.0036), fairness gap halved (0.028 vs 0.054)",
    )
    client.set_model_version_tag(
        REGISTERED_MODEL_NAME, challenger, "promotion_reason",
        "best AUC but widest fairness gap; kept as challenger",
    )
    print(f"@{CHAMPION_ALIAS} -> v{champion} (hgb-07), @{CHALLENGER_ALIAS} -> v{challenger} (logreg-02)")
    for model in list_registered_models():
        print(model["name"], model["aliases"])


if __name__ == "__main__":
    main()
