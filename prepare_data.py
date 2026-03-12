import argparse
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def read_jsonl_from_zip(zip_path: Path, inner_path: str) -> pd.DataFrame:
    """
    Reads a JSONL file from inside a ZIP archive.

    Expected keys at least: id, text
    Adds:
      - base_id: common id across human/machines (everything after first '/')
    """
    with zipfile.ZipFile(zip_path) as z:
        if inner_path not in z.namelist():
            # helpful error
            candidates = [n for n in z.namelist() if n.endswith(".jsonl")][:30]
            raise FileNotFoundError(
                f"File not found in zip: {inner_path}\n"
                f"Some jsonl candidates in zip: {candidates}"
            )
        with z.open(inner_path) as f:
            df = pd.read_json(f, lines=True)

    if "id" not in df.columns or "text" not in df.columns:
        raise ValueError(
            f"Unexpected schema in {inner_path}. "
            f"Columns: {df.columns.tolist()} (need at least: id, text)"
        )

    df = df[["id", "text"]].copy()
    df["id"] = df["id"].astype(str)
    df["text"] = df["text"].astype(str)

    # PAN24: human id and machine id often differ by a prefix before the first "/"
    # Example:
    #   human:    articles-cleaned-truncated/....../art-0001
    #   machine:  gpt-3.5-turbo-0125/....../art-0001
    # We normalize to everything after the first "/"
    df["base_id"] = df["id"].str.split("/", n=1).str[1]
    df["base_id"] = df["base_id"].fillna(df["id"])

    return df


def list_machine_files(zip_path: Path, prefix: str) -> list[str]:
    """List all machine jsonl paths in the zip under a given prefix."""
    with zipfile.ZipFile(zip_path) as z:
        files = [n for n in z.namelist() if n.startswith(prefix) and n.endswith(".jsonl")]
    files = sorted(files)
    if not files:
        raise FileNotFoundError(f"No machine jsonl files found under prefix: {prefix}")
    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=str, required=True, help="Path to pan24-generative-authorship-news.zip")
    parser.add_argument("--out_dir", type=str, default="dataset", help="Output directory for train/dev/test.csv")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--mode",
        choices=["balanced", "all"],
        default="balanced",
        help="balanced: 1 machine example per base_id; all: include all machine versions"
    )

    parser.add_argument("--test_size", type=float, default=0.15, help="Fraction of IDs for test split")
    parser.add_argument("--val_size", type=float, default=0.15, help="Fraction of IDs for validation split")

    parser.add_argument(
        "--machines",
        nargs="*",
        default=None,
        help=(
            "Optional subset of machine file basenames (e.g. gpt-3.5-turbo-0125.jsonl). "
            "If omitted, uses all machine files found."
        ),
    )

    args = parser.parse_args()

    zip_path = Path(args.zip)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)

    root = "pan24-generative-authorship-news"
    human_path = f"{root}/human.jsonl"
    machines_prefix = f"{root}/machines/"

    # --- Load human ---
    human_df = read_jsonl_from_zip(zip_path, human_path)
    human_df["generated"] = 0  # 0 = Human

    # --- Load machine files ---
    machine_files = list_machine_files(zip_path, machines_prefix)

    if args.machines:
        wanted = set(args.machines)
        machine_files = [p for p in machine_files if Path(p).name in wanted]
        if not machine_files:
            raise ValueError(
                "After filtering --machines, no machine files left.\n"
                "Tip: pass basenames exactly as in zip, e.g. gpt-3.5-turbo-0125.jsonl"
            )

    machines = []
    for mf in machine_files:
        df = read_jsonl_from_zip(zip_path, mf)
        df["generated"] = 1  # 1 = Machine/Modified
        df["source_model"] = Path(mf).name.replace(".jsonl", "")
        machines.append(df)

    machines_df = pd.concat(machines, ignore_index=True)

    # --- Use base_id for splitting to prevent leakage ---
    ids = human_df["base_id"].astype(str).unique().tolist()

    # Split IDs: train/val/test
    ids_trainval, ids_test = train_test_split(
        ids, test_size=args.test_size, random_state=args.seed, shuffle=True
    )

    # val_size is fraction of TOTAL, convert to fraction of trainval
    val_frac_of_trainval = args.val_size / (1.0 - args.test_size)
    ids_train, ids_val = train_test_split(
        ids_trainval, test_size=val_frac_of_trainval, random_state=args.seed, shuffle=True
    )

    def build_split(split_ids: list[str]) -> pd.DataFrame:
        # Filter by base_id (not raw id!)
        h = human_df[human_df["base_id"].isin(split_ids)].copy()
        m = machines_df[machines_df["base_id"].isin(split_ids)].copy()

        if args.mode == "balanced":
            # Take exactly 1 random machine example per base_id
            # groupby.sample keeps columns intact, unlike some groupby.apply edge cases
            m = (
                m.groupby("base_id", group_keys=False)
                 .sample(n=1, random_state=args.seed)
                 .reset_index(drop=True)
            )

        # Keep final columns (use base_id as id)
        h = h[["base_id", "text", "generated"]].rename(columns={"base_id": "id"})
        m = m[["base_id", "text", "generated"]].rename(columns={"base_id": "id"})

        out = pd.concat([h, m], ignore_index=True)
        out = out.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
        return out

    train_df = build_split(ids_train)
    val_df = build_split(ids_val)
    test_df = build_split(ids_test)

    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "dev.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)

    print("Saved:")
    print(" -", out_dir / "train.csv", "size:", len(train_df), "class balance:", train_df["generated"].value_counts().to_dict())
    print(" -", out_dir / "dev.csv",   "size:", len(val_df),   "class balance:", val_df["generated"].value_counts().to_dict())
    print(" -", out_dir / "test.csv",  "size:", len(test_df),  "class balance:", test_df["generated"].value_counts().to_dict())
    print("\nMode:", args.mode)
    print("Machines used:", [Path(x).name for x in machine_files])


if __name__ == "__main__":
    main()
