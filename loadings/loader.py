# loadings/loader.py
import os
import glob
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter

def row_to_text(row, index=None, filename=None):
    """
    Convert a dict-like row to a single string with column: value lines.
    """
    parts = []
    for col, val in row.items():
        if pd.isna(val) or val is None or str(val).strip() == "":
            continue
        parts.append(f"{col}: {val}")
    prefix = ""
    if filename:
        prefix += f"Source: {os.path.basename(filename)}\n"
    if index is not None:
        prefix += f"Row: {index}\n"
    return prefix + "\n".join(parts)

def iter_csv_chunks(
    directory="data",
    file_pattern="*.csv",
    max_files=None
):
    """
    Generator that yields (chunks, metadatas) for each row in each CSV.
    Each row is treated as a single chunk. Metadata includes FullName and EmpID (as string).
    """
    file_paths = sorted(glob.glob(os.path.join(directory, file_pattern)))
    if max_files:
        file_paths = file_paths[:max_files]

    for file_path in file_paths:
        # Read as strings to keep formatting consistent (EmpID preserved)
        df = pd.read_csv(file_path, dtype=str).fillna("")
        for idx, series in df.iterrows():
            row_dict = series.to_dict()
            # Ensure EmpID is present and stored as string
            empid_val = str(row_dict.get("EmpID", "")).strip()
            row_dict["EmpID"] = empid_val

            # Build FullName
            firstname = str(row_dict.get("FirstName", "")).strip()
            lastname = str(row_dict.get("LastName", "")).strip()
            full_name = (firstname + " " + lastname).strip()
            row_dict["FullName"] = full_name

            # Build text context for embedding (single chunk per row)
            text = row_to_text(row_dict, index=idx, filename=file_path)
            chunks = [text]
            metadatas = [row_dict]

            yield chunks, metadatas
