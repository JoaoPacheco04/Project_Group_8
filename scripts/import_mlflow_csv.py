from __future__ import annotations

from pathlib import Path

import mlflow
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "artifacts" / "mlflow_all_runs.csv"
TRACKING_URI = f"sqlite:///{(ROOT / 'mlflow_imported.db').as_posix()}"
EXPERIMENT_NAME = "Projeto_Grupo_8_imported_from_csv"


def clean_value(value):
    if pd.isna(value):
        return None
    return value


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    metric_cols = [col for col in df.columns if col.startswith("metric.")]
    param_cols = [col for col in df.columns if col.startswith("param.")]
    tag_cols = [col for col in df.columns if col.startswith("tag.")]

    for _, row in df.iterrows():
        run_name = clean_value(row.get("run_name")) or clean_value(row.get("run_uuid")) or "imported_run"
        with mlflow.start_run(run_name=str(run_name)):
            mlflow.set_tag("imported_from", str(CSV_PATH.relative_to(ROOT)))
            mlflow.set_tag("original_run_uuid", str(clean_value(row.get("run_uuid")) or ""))
            mlflow.set_tag("original_status", str(clean_value(row.get("status")) or ""))
            mlflow.set_tag("original_experiment_name", str(clean_value(row.get("experiment_name")) or ""))

            for col in tag_cols:
                value = clean_value(row.get(col))
                if value is not None:
                    mlflow.set_tag(col.removeprefix("tag."), str(value))

            for col in param_cols:
                value = clean_value(row.get(col))
                if value is not None:
                    mlflow.log_param(col.removeprefix("param."), str(value)[:500])

            for col in metric_cols:
                value = clean_value(row.get(col))
                if value is None:
                    continue
                try:
                    metric_value = float(value)
                except (TypeError, ValueError):
                    continue
                mlflow.log_metric(col.removeprefix("metric."), metric_value)

    print(f"Imported {len(df)} runs into {TRACKING_URI}")
    print("Open with:")
    print("mlflow ui --backend-store-uri sqlite:///mlflow_imported.db --port 5001")


if __name__ == "__main__":
    main()
