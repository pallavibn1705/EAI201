import pandas as pd
import json

# ---------------------------------------------------------------------
# Load Zoo Dataset
# ---------------------------------------------------------------------
def load_zoo_dataset(path):
    df = pd.read_csv(path)
    print("\n=== Loaded Zoo Dataset ===")
    print(df.head())
    return df


# ---------------------------------------------------------------------
# Load Class Dataset
# ---------------------------------------------------------------------
def load_class_dataset(path):
    df = pd.read_csv(path)
    print("\n=== Loaded Class Dataset ===")
    print(df.head())
    return df


# ---------------------------------------------------------------------
# Load JSON Metadata
# ---------------------------------------------------------------------
def load_json_metadata(path):
    with open(path, "r") as f:
        data = json.load(f)
    df = pd.DataFrame(data)

    print("\n=== Loaded JSON Metadata (raw) ===")
    print(df.head())
    return df


# ---------------------------------------------------------------------
# CLEAN JSON METADATA (FIXED VERSION)
# ---------------------------------------------------------------------
def clean_json_metadata(df):
    # Convert all object columns to string safely
    for col in df.columns:
        df[col] = df[col].astype(str).str.lower().str.strip()

    # Standardize column names
    rename_map = {
        "habitats": "habitat",
        "status": "conservation_status",
        "diet_type": "diet",
    }

    df = df.rename(columns=rename_map)

    # Replace 'nan' strings with real NaN
    df = df.replace("nan", pd.NA)

    print("\n=== Cleaned JSON Metadata ===")
    print(df.head())
    return df


# ---------------------------------------------------------------------
# MERGE DATASETS
# ---------------------------------------------------------------------
def merge_all(zoo_df, class_df, json_df):
    # Merge Zoo + JSON on animal_name
    merged = pd.merge(zoo_df, json_df, on="animal_name", how="left")

    # Merge Class info using class_type
    merged = pd.merge(
        merged,
        class_df[["Class_Number", "Class_Type"]],
        left_on="class_type",
        right_on="Class_Number",
        how="left",
    )

    print("\n=== Final Merged Dataset ===")
    print(merged.head())
    return merged


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    zoo_df = load_zoo_dataset("zoo.csv")
    class_df = load_class_dataset("class.csv")
    json_df = load_json_metadata("auxiliary_metadata.json")

    json_df = clean_json_metadata(json_df)

    final_df = merge_all(zoo_df, class_df, json_df)


# Run program
if __name__ == "__main__":
    main()
