from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import UploadFile

from app.core.config import DATA_DIR, UPLOAD_DIR
from app.utils.csv_utils import infer_dataframe_summary, read_csv_flexible
from app.utils.json_utils import make_json_safe

class DataService:
    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = data_dir
        self.upload_dir = UPLOAD_DIR
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    def _slugify(self, name: str) -> str:
        safe = name.lower().strip().replace(" ", "_").replace("-", "_")
        safe = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in safe)
        while "__" in safe:
            safe = safe.replace("__", "_")
        return safe.strip("_") or "dataset"

    def _scan_csv_files(self) -> list[Path]:
        return sorted([p for p in self.data_dir.rglob("*.csv") if p.is_file()])

    def list_datasets(self) -> list[dict]:
        items: list[dict] = []
        for path in self._scan_csv_files():
            try:
                items.append(self.get_dataset_by_path(path))
            except Exception:
                continue
        return items

    from app.utils.json_utils import make_json_safe

    def get_dataset_by_path(self, path: Path) -> dict:
        df = read_csv_flexible(path)
        summary = infer_dataframe_summary(df)
        display_name = path.stem.replace("_", " ")
        payload = {
            "dataset_id": self._slugify(path.stem),
            "filename": path.name,
            "display_name": display_name,
            "path": str(path),
            **summary,
        }
        return make_json_safe(payload)

    def get_dataset_by_id(self, dataset_id: str) -> dict:
        for path in self._scan_csv_files():
            if self._slugify(path.stem) == dataset_id:
                return self.get_dataset_by_path(path)
        raise FileNotFoundError(f"Dataset not found: {dataset_id}")

    async def save_upload(self, file: UploadFile) -> dict:
        original_name = Path(file.filename or "uploaded.csv").name
        suffix = Path(original_name).suffix.lower()
        if suffix != ".csv":
            raise ValueError("Only CSV files are supported")

        stem = Path(original_name).stem
        dataset_id = self._slugify(stem)
        target = self.upload_dir / f"{dataset_id}{suffix}"
        counter = 2
        while target.exists():
            target = self.upload_dir / f"{dataset_id}_{counter}{suffix}"
            counter += 1

        with target.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return self.get_dataset_by_path(target)
