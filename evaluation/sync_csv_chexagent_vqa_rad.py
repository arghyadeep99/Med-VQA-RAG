import pandas as pd
from pathlib import Path

files = ["./filtered_datasets_2/filtered_chexagent_preds_vqa_rad_baseline.csv",
         "./filtered_datasets_2/filtered_chexagent_preds_vqa_rad_graphrag.csv",
         "./filtered_datasets_2/filtered_chexagent_preds_vqa_rad_only_rag.csv"]

key = "q_id"
df = pd.read_csv(files[0])  # start with the first file

# Repeated inner merges drop rows whose id isn't in *all* files
for path in files[1:]:
    df = df.merge(pd.read_csv(path), on=key, how="inner",
                  suffixes=("", f"_{Path(path).stem}"))

df = df.sort_values(key)  # unified, id‑sorted frame
df = df.head(100)    # max 50 values per CSV

# Split back out: keep columns that originated in each file
for path in files:
    stem = Path(path).stem
    suffix = "" if stem == Path(files[0]).stem else f"_{stem}"
    pattern = f"^{key}$|{suffix}$"  # id OR columns with this suffix
    df.filter(regex=pattern).to_csv(f"{stem}_synced.csv", index=False)

print(f"Saved {len(df):,} common rows to *_synced.csv")
