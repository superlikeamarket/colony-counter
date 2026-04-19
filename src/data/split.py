from src.config import SplitConfig
from pathlib import Path
import random
import shutil
from collections import defaultdict
import pandas as pd


# =========================
# IMAGE DISCOVERY
# =========================

def get_image_paths(config: SplitConfig):
    """
    Return a sorted list of all image files in source directory.
    """
    return sorted([
        p for p in config.paths.source_dir.iterdir()
        if p.suffix.lower() in config.image_extensions and p.is_file()
    ])


# =========================
# SPECIES PARSING
# =========================

def extract_species_from_name(filename: str):
    """
    Extract species code from a filename.

    Example:
        sp01_img01.jpg -> sp01
    """
    return filename.split("_")[0]


# =========================
# STRATIFIED SPLIT
# =========================

def stratified_split(
    image_paths: list[Path],
    config: SplitConfig
) -> tuple[list[Path], list[Path], list[Path]]:
    """
    Split image files into train / val / test while preserving species proportions.

    Strategy:
    1. Group images by species
    2. Shuffle images within each species
    3. Split each species group into train / val / test
    4. Merge species-specific splits into global train / val / test lists

    Return:
        train_imgs, val_imgs, test_imgs
    """
    species_grouped = defaultdict(list)

    for img_path in image_paths:
        species = extract_species_from_name(img_path.name)
        species_grouped[species].append(img_path)
    
    train_imgs, val_imgs, test_imgs = [], [], []

    for species, sp_img_paths in species_grouped.items():
        sp_img_paths = sp_img_paths[:]
        random.shuffle(sp_img_paths)

        n = len(sp_img_paths)
        n_train = round(n * config.train_ratio)
        n_val = round(n * config.val_ratio)

        if n_train + n_val > n:
            n_val = n - n_train

        if n >= 3:
            if n_train + n_val == n:
                n_train -= 1
            if n_val == 0:
                n_val = 1
                n_train -= 1

        species_train = sp_img_paths[:n_train]
        species_val = sp_img_paths[n_train:n_train + n_val]
        species_test = sp_img_paths[n_train + n_val:]

        train_imgs.extend(species_train)
        val_imgs.extend(species_val)
        test_imgs.extend(species_test)

        print(
            f"{species}: total={n}, "
            f"train={len(species_train)}, val={len(species_val)}, test={len(species_test)}"
        )

    random.shuffle(train_imgs)
    random.shuffle(val_imgs)
    random.shuffle(test_imgs)

    return train_imgs, val_imgs, test_imgs


# =========================
# COPYING RELATED FILES
# =========================

def copy_related_files(image_paths, split_name, config: SplitConfig):
    """
    Copy images and matching annotation files into the appropriate split folders.

    For each image:
    - copy the image into images/<split_name>/
    - if a matching .txt exists, copy it into labels/<split_name>/
    - if a matching .xml exists, copy it into annotations_xml/<split_name>/
    """
    base_dir = config.paths.output_dir

    images_dir = base_dir / "images" / split_name
    labels_dir = base_dir / "labels" / split_name
    xml_dir = base_dir / "annotations_xml" / split_name

    for img_path in image_paths:
        stem = img_path.stem

        shutil.copy2(img_path, images_dir / img_path.name)

        txt_path = config.paths.source_dir / f"{stem}.txt"
        if txt_path.exists():
            shutil.copy2(txt_path, labels_dir / txt_path.name)

        xml_path = config.paths.source_dir / f"{stem}.xml"
        if xml_path.exists():
            shutil.copy2(xml_path, xml_dir / xml_path.name)


# =========================
# SAVE SPLIT SUMMARY CSV
# =========================

def save_split_csv(image_paths, split_name, config: SplitConfig):
    """
    Save a CSV file listing the images in one split.

    Suggested columns:
    - filename
    - basename
    - species
    - split

    Output:
    output_dir / "splits" / f"{split_name}.csv"
    """
    df = pd.DataFrame({
        "filename": [p.name for p in image_paths],
        "basename": [p.stem for p in image_paths],
        "species": [extract_species_from_name(p.name) for p in image_paths],
        "split": split_name
    })
    df.to_csv(config.paths.output_dir / "splits" / f"{split_name}.csv", index=False)


# =========================
# METADATA TABLE SPLITTING
# =========================

def split_metadata_table(
        table_path,
        train_names,
        val_names,
        test_names,
        config: SplitConfig
    ):
    """
    Split a metadata table into train / val / test versions based on filename
        membership.

    Supported table types:
    - .csv
    - .tsv
    - .xls / .xlsx

    Steps:
    1. Check if the file exists
    2. Load it with pandas
    3. Check that FILENAME_COLUMN exists
    4. Create a helper column with normalized filename only
    5. Filter rows into train / val / test
    6. Save each split table in output_dir / "splits"/
    """
    table_path = Path(table_path)
    if not table_path.exists():
        print(f"Skipping missing table: {table_path}")
        return
    
    if table_path.suffix.lower() == ".csv":
        df = pd.read_csv(table_path)
    elif table_path.suffix.lower() == ".tsv":
        df = pd.read_csv(table_path, sep="\t")
    elif table_path.suffix.lower() in [".xls", ".xlsx"]:
        df = pd.read_excel(table_path)
    else:
        print(f"Unsupported table format: {table_path}")
        return
    
    if config.filename_column not in df.columns:
        print(f"Column '{config.filename_column}' not found in {table_path.name}")
        print(f"Columns are: {list(df.columns)}")
        return
    
    df["_match_name"] = df[config.filename_column].astype(str).apply(lambda x: Path(x).name)

    train_df = df[df["_match_name"].isin(train_names)].drop(columns=["_match_name"])
    val_df = df[df["_match_name"].isin(val_names)].drop(columns=["_match_name"])
    test_df = df[df["_match_name"].isin(test_names)].drop(columns=["_match_name"])

    out_base = table_path.stem
    out_ext = table_path.suffix.lower()

    if out_ext == ".csv":
        train_df.to_csv(config.paths.output_dir / "splits" / f"{out_base}_train.csv", index=False)
        val_df.to_csv(config.paths.output_dir / "splits" / f"{out_base}_val.csv", index=False)
        test_df.to_csv(config.paths.output_dir / "splits" / f"{out_base}_test.csv", index=False)
    elif out_ext == ".tsv":
        train_df.to_csv(config.paths.output_dir / "splits" / f"{out_base}_train.tsv", sep="\t", index=False)
        val_df.to_csv(config.paths.output_dir / "splits" / f"{out_base}_val.tsv", sep="\t", index=False)
        test_df.to_csv(config.paths.output_dir / "splits" / f"{out_base}_test.tsv", sep="\t", index=False)
    elif out_ext in [".xls", ".xlsx"]:
        train_df.to_excel(config.paths.output_dir / "splits" / f"{out_base}_train.xlsx", index=False)
        val_df.to_excel(config.paths.output_dir / "splits" / f"{out_base}_val.xlsx", index=False)
        test_df.to_excel(config.paths.output_dir / "splits" / f"{out_base}_test.xlsx", index=False)

    print(f"Split metadata table: {table_path.name}")


# =========================
# REPORTING / DEBUGGING
# =========================

def print_species_distribution(image_paths, split_name):
    """
    Print how many images of each species ended up in one split.

    Example output:
        TRAIN species distribution:
          sp01: 8
          sp02: 8
          sp03: 7
    """
    counts = defaultdict(int)
    for p in image_paths:
        species = extract_species_from_name(p.name)
        counts[species] += 1

    print(f"\n{split_name.upper()} species distribution:")
    for species in sorted(counts):
        print(f"  {species}: {counts[species]}")


# =========================
# MAIN
# =========================

def main():
    """
    Main workflow:

    1. Validate that split ratios sum to 1.0
    2. Set random seed
    3. Create output directories
    4. Get image files
    5. Run stratified split
    6. Print summary sizes
    7. Print per-species distribution
    8. Copy images and annotations
    9. Save split summary CSVs
    10. Build filename sets for train/val/test
    11. Split any metadata tables listed in table_files
    """
    config = SplitConfig()
    
    assert abs(config.train_ratio + config.val_ratio + config.test_ratio - 1.0) < 1e-6, \
        "TRAIN_RATIO + VAL_RATIO + TEST_RATIO must equal 1.0"
    random.seed(config.random_seed)

    # Step 1: create folders
    config.ensure_split_dirs()

    # Step 2: get images and species names
    image_paths = get_image_paths(config)
    print(f"Found {len(image_paths)} images")
    species_names = {extract_species_from_name(p.name) for p in image_paths}
    print(f"Found {len(species_names)} species\n")

    # Step 3: stratified split
    train_imgs, val_imgs, test_imgs = stratified_split(image_paths, config)

    # Step 4: print final counts
    print(f"\nFinal split sizes:")
    print(f"Train: {len(train_imgs)}")
    print(f"Val:   {len(val_imgs)}")
    print(f"Test:  {len(test_imgs)}")

    # Step 5: print species distribution for each split
    print_species_distribution(train_imgs, "train")
    print_species_distribution(val_imgs, "val")
    print_species_distribution(test_imgs, "test")

    # Step 6: copy files for each split
    copy_related_files(train_imgs, "train", config)
    copy_related_files(val_imgs, "val", config)
    copy_related_files(test_imgs, "test", config)

    # Step 7: save train/val/test CSV summaries
    save_split_csv(train_imgs, "train", config)
    save_split_csv(val_imgs, "val", config)
    save_split_csv(test_imgs, "test", config)

    # Step 8: create sets of filenames for metadata matching
    train_names = {p.name for p in train_imgs}
    val_names = {p.name for p in val_imgs}
    test_names = {p.name for p in test_imgs}

    # Step 9: split metadata tables
    for table_file in config.table_files:
        split_metadata_table(
            config.paths.source_dir / table_file,
            train_names,
            val_names,
            test_names,
            config
        )
    

if __name__ == "__main__":
    main()