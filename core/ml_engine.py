import pandas as pd
import numpy as np


def load_dataset(file_path):
    try:
        df = pd.read_csv(file_path)
    except pd.errors.ParserError:
        df = pd.read_csv(
            file_path,
            engine="python",
            on_bad_lines="warn"
        )

    return df

def detect_target_column(df):
    """
    Automatically detect the most likely target column.
    """

    possible_targets = [
        "target",
        "label",
        "class",
        "outcome",
        "dropout",
        "status",
        "y",
        "result",
        "final_result",
    ]

    # First: look for common target names
    for column in df.columns:
        if column.lower().strip() in possible_targets:
            print("Target column detected:", column)
            return column

    # Second: look for columns with relatively few unique values
    candidates = []

    for column in df.columns:
        unique_count = df[column].nunique()

        if 2 <= unique_count <= 10:
            candidates.append((column, unique_count))

    if len(candidates) == 1:
        target = candidates[0][0]
        print("Target column detected:", target)
        return target

    print("\nPossible target columns:")
    for column, count in candidates:
        print(f"- {column}: {count} unique values")

    raise ValueError(
        "Could not automatically determine the target column."
    )