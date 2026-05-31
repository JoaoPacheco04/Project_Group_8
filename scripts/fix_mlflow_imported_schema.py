from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "mlflow_imported.db"
BACKUP_PATH = ROOT / "mlflow_imported_before_stamp_fix.db"
EXPECTED_VERSION = "1b5f0d9ad7c1"


def main() -> None:
    if not DB_PATH.exists():
        raise FileNotFoundError(DB_PATH)

    if not BACKUP_PATH.exists():
        shutil.copy2(DB_PATH, BACKUP_PATH)

    con = sqlite3.connect(DB_PATH)
    con.execute("update alembic_version set version_num = ?", (EXPECTED_VERSION,))
    con.commit()
    version = con.execute("select version_num from alembic_version").fetchall()
    runs = con.execute("select count(1) from runs").fetchone()[0]
    con.close()

    print(f"Updated {DB_PATH.name}: version={version}, runs={runs}")


if __name__ == "__main__":
    main()
