import os
import pandas as pd


def csv_to_text_chunks(
        input_csv: str,
        output_dir: str,
        columns: list[str],
        chunk_size: int = 100
) -> None:

    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Use pandas to stream through the CSV in chunks
    for chunk_idx, chunk in enumerate(
            pd.read_csv(input_csv, usecols=columns, chunksize=chunk_size)
    ):
        # Name each file sequentially
        out_path = os.path.join(output_dir, f'chunk_{chunk_idx + 1}.txt')

        with open(out_path, 'w', encoding='utf-8') as out_file:
            # Iterate rows in this chunk
            for _, row in chunk.iterrows():
                # Build "col1: val1, col2: val2, ..." for this row
                line = ', '.join(f"{col}: {row[col]}" for col in columns)
                out_file.write(line + '\n')

        print(f"Wrote {len(chunk)} rows to {out_path}")


if __name__ == "__main__":
    # Example usage:
    INPUT_CSV = "./rexgradient_val.csv"
    OUTPUT_DIR = "./txt_data"
    # List only the columns you want to include
    COLUMNS = ["PatientSex", "PatientAge", "StudyDescription", "Indication", "Comparison", "Findings", "Impression"]
    CHUNK_SIZE = 50  # rows per file

    csv_to_text_chunks(INPUT_CSV, OUTPUT_DIR, COLUMNS, CHUNK_SIZE)
