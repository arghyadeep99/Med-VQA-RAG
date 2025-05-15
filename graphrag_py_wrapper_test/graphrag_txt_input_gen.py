import os
import math
import pandas as pd


def csv_to_random_text_chunks(
        input_csv: str,
        output_dir: str,
        columns: list[str],
        num_files: int = 20,
        seed: int | None = 42
) -> None:

    # load and shuffle
    df = pd.read_csv(input_csv, usecols=columns)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df = df[:int(0.1 * len(df))]

    # compute chunk size so that we get exactly num_files
    total_rows = len(df)
    chunk_size = math.ceil(total_rows / num_files)

    os.makedirs(output_dir, exist_ok=True)

    for i in range(num_files):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, total_rows)
        chunk = df.iloc[start:end]

        out_path = os.path.join(output_dir, f'chunk_{i+1}.txt')
        with open(out_path, 'w', encoding='utf-8') as out_file:
            for _, row in chunk.iterrows():
                line = ', '.join(f"{col}: {row[col]}" for col in columns)
                out_file.write(line + '\n')

        print(f"Wrote {len(chunk)} rows to {out_path}")


if __name__ == "__main__":
    INPUT_CSV = "./rexgradient_val.csv"
    OUTPUT_DIR = "txt_data"
    COLUMNS = [
        "PatientSex", "PatientAge", "StudyDescription",
        "Indication", "Comparison", "Findings", "Impression"
    ]
    NUM_FILES = 10  # produce exactly 10 text files

    csv_to_random_text_chunks(INPUT_CSV, OUTPUT_DIR, COLUMNS, NUM_FILES)
