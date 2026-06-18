import json
import glob
import os
import pandas as pd
from pathlib import Path


def batch_check_deep_folders(base_folder_path, csv_file_path, csv_column_name="성분명1"):
    if not os.path.isdir(base_folder_path):
        print(f" Base folder not found: {base_folder_path}")
        return

    search_pattern = os.path.join(base_folder_path, "**", "*.json")
    json_files = glob.glob(search_pattern, recursive=True)

    if not json_files:
        print(f" No .json files found in '{base_folder_path}' or its subfolders.")
        return

    print(f" Scanning directories... Found a total of {len(json_files)} JSON files across all categories.")

    json_ingredients = set()
    parsed_count = 0

    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)

            images = json_data.get("images", [])
            for img in images:
                material = img.get("dl_material")
                if material:
                    json_ingredients.add(str(material).strip())
            parsed_count += 1
        except Exception as e:
            print(f" Error reading file [{os.path.relpath(file_path, base_folder_path)}]: {e}")

    print(f" JSON parsing completed: Successfully parsed {parsed_count}/{len(json_files)} files.")
    print(f"A total of {len(json_ingredients)} unique drug ingredients were successfully extracted across multiple folders.")

    if not os.path.exists(csv_file_path):
        print(f" File not found: {csv_file_path}")
        return

    df = None
    file_path_str = str(csv_file_path)
    if file_path_str.endswith('.xlsx') or file_path_str.endswith('.xls'):
        try:
            df = pd.read_excel(file_path_str, engine='openpyxl')
        except Exception as e:
            print(f" Failed to read Excel file: {e}")
            return
    else:
        for encoding in ['utf-8', 'cp949', 'utf-8-sig', 'euc-kr']:
            try:
                df = pd.read_csv(csv_file_path, encoding=encoding)
                break
            except (UnicodeDecodeError, Exception):
                continue

    if df is None:
        print(f" Failed to read file. Please verify the file format and encoding.")
        return

    if csv_column_name not in df.columns:
        print(f" Column '{csv_column_name}' not found in CSV.")
        print(f" Available columns in CSV: {list(df.columns)}")
        return

    csv_ingredients = set(df[csv_column_name].dropna().astype(str).str.strip())
    print(f" CSV loaded successfully. Extracted {len(csv_ingredients)} unique ingredients from column '{csv_column_name}'.")

    matched = json_ingredients.intersection(csv_ingredients)
    only_in_json = json_ingredients - csv_ingredients
    only_in_csv = csv_ingredients - json_ingredients

    print("\n" + "=" * 25 + " Cross-Folder Comparison Report " + "=" * 25)
    print(f" Number of perfectly matched ingredients: {len(matched)}")

    print(f"\n Found only in JSON (Missing in CSV): {len(only_in_json)}")
    if only_in_json:
        print(f"   Discrepancy Details: {list(only_in_json)}")

    print(f"\n Found only in CSV (Missing in JSON): {len(only_in_csv)}")
    if only_in_csv:
        csv_show = list(only_in_csv)[:10] if len(only_in_csv) > 10 else list(only_in_csv)
        print(f"   CSV Sample Details (Top 10): {csv_show} ...")
    print("=" * 75)

    if only_in_json:
        error_df = pd.DataFrame(list(only_in_json), columns=["Missing_In_CSV"])
        error_df.to_csv("missing_in_csv.csv", index=False, encoding="utf-8-sig")
        print(" Saved the list of missing ingredients to local file: missing_in_csv.csv")


if __name__ == "__main__":
    main_file = Path(__file__).resolve()
    src_dir = main_file.parent
    PROJECT_ROOT = src_dir.parent

    csv_path = PROJECT_ROOT / "data" / "contraindicated_drugs.xlsx"
    CSV_COL = "material_1"

    batch_check_deep_folders(PROJECT_ROOT, csv_path, CSV_COL)