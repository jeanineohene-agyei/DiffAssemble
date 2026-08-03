from pathlib import Path
import zipfile
import shutil

ZIP_DIR = Path("/data/batches")
DEST_DIR = Path("/data/jeanine/processed_data")

DEST_DIR.mkdir(parents=True, exist_ok=True)

for zip_path in sorted(ZIP_DIR.glob("*.zip")):
    batch_name = zip_path.stem.split("_")[-1]  # e.g. "071"
    batch_dir = ZIP_DIR / f"batch_{batch_name}"

    print(f"Processing {zip_path.name}")

    if batch_dir.exists():
        shutil.rmtree(batch_dir)
    batch_dir.mkdir()

    try:
        # Extract into batch_###
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(batch_dir)

        dest = DEST_DIR / batch_dir.name

        if dest.exists():
            print(f"  {dest.name} already exists. Skipping.")
            shutil.rmtree(batch_dir)
            continue

        # Move the entire batch folder
        shutil.move(str(batch_dir), str(dest))

        # Delete the zip
        zip_path.unlink()

        print(f"  Moved to {dest}")
        print(f"  Deleted {zip_path.name}")

    except Exception as e:
        print(f"  ERROR: {e}")
        if batch_dir.exists():
            shutil.rmtree(batch_dir)

print("Done.")