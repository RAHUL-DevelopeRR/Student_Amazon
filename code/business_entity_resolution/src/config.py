from pathlib import Path
import csv
import json
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[3]
SEED = 42


def connect(path=None, memory="512MB"):
    import duckdb
    path = Path(path or ROOT / "artifacts/audit.duckdb")
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path))
    con.execute("SET memory_limit = ?", [memory])
    con.execute("SET threads = 1")
    con.execute("SET preserve_insertion_order = false")
    return con


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)


def log_experiment(stage, metrics, parameters):
    path = ROOT / "experiments/experiment_log.csv"
    path.parent.mkdir(exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["utc", "stage", "parameters_json", "metrics_json"])
        w.writerow([datetime.now(timezone.utc).isoformat(), stage,
                    json.dumps(parameters, sort_keys=True), json.dumps(metrics, sort_keys=True)])
