"""
Renumber clips in kicking/ punching/ shooting/ folders to 1.mp4, 2.mp4, ...
Order = file modified time (oldest first).

Safety:
- Two-phase rename (temp names first) so no file ever gets overwritten
  mid-run even if a target number already exists.
- Writes a rename_map_<folder>.csv (old_name -> new_name) next to each
  folder BEFORE renaming, so original names are never lost.
- Dry run by default. Pass --apply to actually rename.

Usage:
    python rename_clips.py            # dry run, just prints + writes CSVs
    python rename_clips.py --apply    # actually renames files
"""
import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).parent
FOLDERS = ["kicking", "punching", "shooting"]

# portrait subfolder name differs per action folder
PORTRAIT_SUBDIRS = {
    "kicking": "Potrait",
    "punching": "Potrait",
    "shooting": "Portrait_clips",
}


def collect(folder: Path):
    files = [f for f in folder.iterdir() if f.is_file()]
    files.sort(key=lambda f: f.stat().st_mtime)
    return files


def collect_portrait(folder: Path):
    # filenames are VID_<date>_<time>.mp4 -> sorting by name == chronological
    files = [f for f in folder.iterdir() if f.is_file()]
    files.sort(key=lambda f: f.name)
    return files


def rename_pass(folder: Path, mapping, csv_path, apply: bool, label: str):
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["old_name", "new_name"])
        w.writerows(mapping)

    print(f"\n=== {label}  ({len(mapping)} files) ===")
    for old, new in mapping[:5]:
        print(f"  {old}  ->  {new}")
    if len(mapping) > 5:
        print(f"  ... ({len(mapping) - 5} more, see {csv_path.name})")

    if not apply:
        return

    files = [folder / old for old, _ in mapping]
    temp_paths = []
    for f, (old, new) in zip(files, mapping):
        tmp = folder / f"__tmp__{f.name}__{new}"
        f.rename(tmp)
        temp_paths.append((tmp, new))
    for tmp, new in temp_paths:
        tmp.rename(folder / new)
    print(f"  renamed {len(mapping)} files in {label}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="actually rename (default: dry run)")
    ap.add_argument("--skip-main", action="store_true", help="don't touch main folder files, only portrait subfolders")
    args = ap.parse_args()

    for name in FOLDERS:
        folder = ROOT / name
        if not folder.is_dir():
            print(f"[skip] {name}/ not found")
            continue

        if not args.skip_main:
            files = collect(folder)
            mapping = [(f.name, f"{i}{f.suffix.lower()}") for i, f in enumerate(files, start=1)]
            rename_pass(folder, mapping, ROOT / f"rename_map_{name}.csv", args.apply, f"{name}/")

        # --- portrait subfolder: potrait_01.mp4, potrait_02.mp4, ... ---
        sub_name = PORTRAIT_SUBDIRS.get(name)
        sub_folder = folder / sub_name if sub_name else None
        if sub_folder and sub_folder.is_dir():
            sub_files = collect_portrait(sub_folder)
            sub_mapping = [
                (f.name, f"potrait_{i:02d}{f.suffix.lower()}")
                for i, f in enumerate(sub_files, start=1)
            ]
            rename_pass(
                sub_folder,
                sub_mapping,
                ROOT / f"rename_map_{name}_portrait.csv",
                args.apply,
                f"{name}/{sub_name}/",
            )

    if not args.apply:
        print("\nDry run only. Review rename_map_*.csv, then re-run with --apply.")


if __name__ == "__main__":
    main()
