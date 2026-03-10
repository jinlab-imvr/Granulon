"""
SEED-Bench Dataset Processing
"""

from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
from typing import Iterable, List, Dict, Optional
from tqdm import tqdm


def download_seed_bench(dest_dir: str, repo_id: str = "lmms-lab/SEED-Bench") -> Path:
    try:
        from huggingface_hub import snapshot_download
    except Exception as e:
        raise RuntimeError(
            "huggingface_hub is required for download; install it or skip download"
        ) from e

    dest = Path(dest_dir).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(dest),
        revision="main",
        max_workers=4,
        allow_patterns=["*.parquet"],
    )
    return Path(snapshot_path)


def bytes_to_base64(b: bytes) -> str:
    return base64.b64encode(b).decode("utf-8")


QUESTION_TEMPLATE = (
    "{question}"
    "\nOptions:"
    "\nA. {choice_a}"
    "\nB. {choice_b}"
    "\nC. {choice_c}"
    "\nD. {choice_d}"
    "\n(Answer with the option content)"
)


def ensure_image_marker(q: str) -> str:
    q = q.rstrip("\n")
    if not q.endswith("<image>"):
        q = f"{q}\n<image>"
    return q


def pick_answer_text(row: Dict) -> str:
    ans = (row.get("answer") or "").strip()
    mapping = {
        "A": row.get("choice_a"),
        "B": row.get("choice_b"),
        "C": row.get("choice_c"),
        "D": row.get("choice_d"),
    }
    if ans in mapping and mapping[ans] is not None:
        return str(mapping[ans])
    return str(ans)


def extract_rel_image_name(image_field) -> Optional[str]:
    try:
        if image_field is None:
            return None
        if isinstance(image_field, (list, tuple)):
            elem = image_field[0] if image_field else None
        else:
            elem = image_field[0]
        if isinstance(elem, dict):
            p = elem.get("path")
            if p:
                base = os.path.basename(str(p))
                if not base.lower().endswith((".jpg", ".jpeg", ".png")):
                    base = base + ".jpg"
                return base
    except Exception:
        pass
    return None


def maybe_write_local_image(image_field, images_dir: Path, rel_name: str) -> Optional[Path]:
    try:
        if image_field is None:
            return None
        if isinstance(image_field, (list, tuple)):
            elem = image_field[0] if image_field else None
        else:
            elem = image_field[0]
        if isinstance(elem, dict) and elem.get("bytes"):
            images_dir.mkdir(parents=True, exist_ok=True)
            out_path = images_dir.joinpath(rel_name)
            with open(out_path, "wb") as f:
                f.write(elem["bytes"])
            return out_path
    except Exception:
        return None
    return None


def rows_from_parquets(parquet_paths: Iterable[Path], limit: Optional[int] = None) -> List[Dict]:
    import pandas as pd
    records: List[Dict] = []
    n = 0
    for p in parquet_paths:
        df = pd.read_parquet(str(p), engine="pyarrow")
        for _, row in df.iterrows():
            rec = {k: row[k] for k in df.columns}
            records.append(rec)
            n += 1
            if limit is not None and n >= limit:
                return records
    return records


def normalize_seed_records(records: List[Dict], images_subdir: str = "images", write_local_images: bool = False, out_root: Optional[Path] = None) -> List[Dict]:
    out: List[Dict] = []
    images_dir = out_root.joinpath(images_subdir) if (write_local_images and out_root) else None

    for row in tqdm(records, desc="Normalizing SEED-Bench records", unit="record"):
        q = QUESTION_TEMPLATE.format(
            question=str(row.get("question", "")).strip(),
            choice_a=str(row.get("choice_a", "")),
            choice_b=str(row.get("choice_b", "")),
            choice_c=str(row.get("choice_c", "")),
            choice_d=str(row.get("choice_d", "")),
        )
        q = ensure_image_marker(q)

        a_text = pick_answer_text(row)

        rel_img_name = extract_rel_image_name(row.get("image"))
        if rel_img_name is None:
            did = str(row.get("data_id", "seed_image"))
            rel_img_name = f"{os.path.basename(did)}.jpg"

        if images_dir is not None:
            maybe_write_local_image(row.get("image"), images_dir, rel_img_name)

        rid = f"SEED_{row.get('data_id', '')}_{row.get('question_id', '')}"

        item = {
            "id": rid,
            "image": rel_img_name, 
            "conversations": [
                {"from": "human", "value": q},
                {"from": "gpt", "value": str(a_text)},
            ],
        }
        out.append(item)

    return out


def save_chat_json(items: List[Dict], out_root: Path, filename: str = "chat.json") -> Path:
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root.joinpath(filename)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=4)
    return out_path


def _collect_parquet_paths_from_dir(root: Path) -> List[Path]:
    return sorted(root.rglob("*.parquet"))


def main():
    parser = argparse.ArgumentParser(description="Download and normalize SEED-Bench to LLaVA style chat.json")
    parser.add_argument("--repo-id", type=str, default="lmms-lab/SEED-Bench", help="HuggingFace dataset repo id")
    parser.add_argument("--raw-dir", type=str, default="./raw_data", help="Directory to download/store raw data")
    parser.add_argument("--out-dir", type=str, default="./datasets/seed", help="Output dataset root directory (relative path recommended, e.g., datasets/seed)")
    parser.add_argument("--no-write-images", action="store_true", help="Do not write image files (only keep relative path names)")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_root = Path(args.out_dir)

    parquet_paths: List[Path] = _collect_parquet_paths_from_dir(raw_dir) if raw_dir.exists() else []
    if not parquet_paths:
        print(f"No parquet files found in {raw_dir}, downloading from {args.repo_id}...")
        from datasets import load_dataset
        ds = load_dataset("lmms-lab/SEED-Bench")
        parquet_paths = _collect_parquet_paths_from_dir(raw_dir)

    if not parquet_paths:
        raise SystemExit(f"No parquet files found in {raw_dir} after download")

    records = rows_from_parquets(parquet_paths, limit=None)
    items = normalize_seed_records(
        records, images_subdir="images", write_local_images=(not args.no_write_images), out_root=out_root
    )
    chat_path = save_chat_json(items, out_root)

    print(json.dumps({
        "raw_dir": str(Path(raw_dir).resolve()),
        "out_root": str(out_root.resolve()),
        "chat_json": str(chat_path.resolve()),
        "num_items": len(items),
        "example": items[0] if items else None,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()