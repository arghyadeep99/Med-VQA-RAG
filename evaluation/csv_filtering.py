import os
import argparse
import pandas as pd
from pathlib import Path
import tiktoken
import logging
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s │ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# env setup
env_path = "./.env"
if os.path.exists(env_path):
    load_dotenv(env_path)
    logger.info(f"Loaded environment from {env_path}")
else:
    logger.warning(f".env file not found at {env_path}")


enc = tiktoken.get_encoding("cl100k_base")  # GPT‑4o‑mini’s encoding


def n_tokens(text: str) -> int:
    return len(enc.encode(text))


def filter_csv(infile: str, outfile: str, cols: list[str], limit: int):
    df = pd.read_csv(infile)
    # Concatenate the chosen columns row‑wise, handle NaNs
    joined = df[cols].fillna("").agg(" ".join, axis=1)
    mask = joined.apply(n_tokens) <= limit  # keep rows within budget

    kept, removed = df[mask], df[~mask]
    kept.to_csv(outfile, index=False)
    if not removed.empty:  # optional audit trail
        rm_path = Path(outfile).with_stem(Path(outfile).stem + "_removed")
        removed.to_csv(rm_path, index=False)

    print(f"Kept {len(kept)} rows; removed {len(removed)} that exceeded {limit} tokens.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", required=True, help="source CSV")
    ap.add_argument("--outfile", required=True, help="destination CSV")
    ap.add_argument("--columns", required=True,
                    help="comma‑separated list of columns to concatenate")
    ap.add_argument("--limit", type=int, default=8000)
    args = ap.parse_args()

    cols = [c.strip() for c in args.columns.split(",")]
    filter_csv(args.infile, args.outfile, cols, args.limit)
