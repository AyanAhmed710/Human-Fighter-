"""
Build empty labeling CSV for all clips (main + portrait subfolders).
clip_id = path relative to project root. action pre-filled from folder
name (edit if wrong). camera_angle / limb_used / participant_id left
blank for manual fill.

Usage: python make_label_template.py
"""
import csv
from pathlib import Path

ROOT = Path(__file__).parent
FOLDERS = ["kicking", "punching", "shooting"]
PORTRAIT_SUBDIRS = {
    "kicking": "Potrait",
    "punching": "Potrait",
    "shooting": "Portrait_clips",
}

COLUMNS = ["clip_id", "action", "camera_angle", "participant_id", "notes"]


def main():
    rows = []
    for name in FOLDERS:
        folder = ROOT / name
        if not folder.is_dir():
            continue

        main_files = sorted(
            (f for f in folder.iterdir() if f.is_file()),
            key=lambda f: int(f.stem) if f.stem.isdigit() else 0,
        )
        for f in main_files:
            rows.append([f"{name}/{f.name}", name, "", "", "", ""])

        sub_name = PORTRAIT_SUBDIRS.get(name)
        sub_folder = folder / sub_name if sub_name else None
        if sub_folder and sub_folder.is_dir():
            sub_files = sorted(f for f in sub_folder.iterdir() if f.is_file())
            for f in sub_files:
                rows.append([f"{name}/{sub_name}/{f.name}", name, "", "", "", ""])

    out_path = ROOT / "labels_template.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(COLUMNS)
        w.writerows(rows)

    print(f"wrote {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
