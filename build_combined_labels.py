"""
Combine kicking/punching/shooting label sources into one dataset CSV.

Sources:
  - kicking_labels.pdf   (hand-transcribed from notebook photos -> hardcoded below)
  - punching_labels.xlsx (ID, Label)   F/R/L
  - shooting_labels.xlsx (ID, Symbol)  F/R/L

Symbol mapping: F -> front, R -> side_right, L -> side_left

Known source issues (kept as flagged rows, not silently dropped):
  - kicking ID 1: not present in kicking_labels.pdf (missing from source)
  - kicking IDs 2-63: transcribed from an angled photo per source note -> verify
  - punching ID 48: crossed out in notebook / excluded per source note
  - shooting ID 107: not present in shooting_labels.xlsx (missing from source)

Also appends portrait-subfolder clips (unlabeled, camera_angle left blank)
so the combined file is a complete manifest for participant_id entry.

Output: combined_labels.csv with columns:
  clip_id, action, camera_angle, participant_id, notes
"""
import csv
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parent

SYMBOL_MAP = {"F": "front", "R": "side_right", "L": "side_left"}

PORTRAIT_SUBDIRS = {
    "kicking": "Potrait",
    "punching": "Potrait",
    "shooting": "Portrait_clips",
}

# --- kicking: hand-transcribed from kicking_labels.pdf (IDs 2-211) ---
KICKING_RAW = """2 F 3 R 4 F 5 L 6 R 7 F 8 L 9 R 10 F 11 L 12 R 13 R 14 F 15 L 16 R 17 F 18 L 19 L
20 R 21 L 22 L 23 R 24 R 25 R 26 R 27 R 28 R 29 L 30 L 31 L 32 L 33 R 34 L 35 L 36 L
37 F 38 F 39 R 40 R 41 F 42 L 43 L 44 R 45 R 46 R 47 R 48 F 49 L 50 L 51 R 52 R 53 L
54 L 55 L 56 R 57 L 58 F 59 L 60 L 61 R 62 L 63 R 64 F 65 F 66 R 67 F 68 R 69 F 70 F
71 F 72 F 73 R 74 F 75 F 76 R 77 R 78 R 79 R 80 R 81 L 82 L 83 L 84 L 85 L 86 L 87 R
88 L 89 L 90 L 91 F 92 L 93 L 94 L 95 F 96 R 97 R 98 R 99 F 100 L 101 R 102 F 103 L
104 L 105 F 106 F 107 R 108 F 109 F 110 R 111 F 112 F 113 R 114 L 115 R 116 L 117 L
118 L 119 L 120 L 121 L 122 L 123 F 124 F 125 R 126 F 127 L 128 L 129 R 130 L 131 F
132 F 133 F 134 R 135 R 136 F 137 R 138 R 139 R 140 L 141 F 142 F 143 F 144 L 145 F
146 R 147 R 148 R 149 R 150 L 151 R 152 L 153 R 154 R 155 L 156 F 157 F 158 L 159 F
160 L 161 R 162 L 163 L 164 L 165 F 166 L 167 R 168 F 169 R 170 L 171 R 172 L 173 L
174 R 175 L 176 F 177 R 178 R 179 L 180 R 181 F 182 F 183 F 184 F 185 L 186 R 187 R
188 L 189 L 190 L 191 F 192 F 193 R 194 F 195 L 196 L 197 R 198 R 199 R 200 L 201 F
202 F 203 L 204 L 205 F 206 R 207 L 208 L 209 L 210 R 211 R"""


def parse_kicking():
    toks = KICKING_RAW.split()
    pairs = list(zip(toks[0::2], toks[1::2]))
    data = {int(i): sym for i, sym in pairs}

    rows = []
    for i in range(1, 212):
        if i not in data:
            rows.append((f"kicking/{i}.mp4", "kicking", "", "not labeled (missing from source pdf)"))
            continue
        angle = SYMBOL_MAP[data[i]]
        note = "verify vs notebook (angled-photo transcription)" if 2 <= i <= 63 else ""
        rows.append((f"kicking/{i}.mp4", "kicking", angle, note))
    return rows


def parse_xlsx(path, action, id_col, label_col, excluded_ids, n_clips):
    df = pd.read_excel(path)
    df = df[df[id_col].apply(lambda v: str(v).strip().isdigit())].copy()
    df[id_col] = df[id_col].astype(int)
    lookup = dict(zip(df[id_col], df[label_col]))

    rows = []
    for i in range(1, n_clips + 1):
        clip_id = f"{action}/{i}.mp4"
        if i in excluded_ids:
            rows.append((clip_id, action, "", "excluded - crossed out in notebook"))
        elif i not in lookup:
            rows.append((clip_id, action, "", "not labeled (missing from source)"))
        else:
            rows.append((clip_id, action, SYMBOL_MAP[lookup[i]], ""))
    return rows


def portrait_rows():
    rows = []
    for action, sub_name in PORTRAIT_SUBDIRS.items():
        sub_folder = ROOT / action / sub_name
        if not sub_folder.is_dir():
            continue
        for f in sorted(f for f in sub_folder.iterdir() if f.is_file()):
            rows.append((f"{action}/{sub_name}/{f.name}", action, "", "portrait clip - not labeled"))
    return rows


def main():
    rows = []
    rows += parse_kicking()
    rows += parse_xlsx(ROOT / "punching_labels.xlsx", "punching", "ID", "Label", excluded_ids={48}, n_clips=221)
    rows += parse_xlsx(ROOT / "shooting_labels.xlsx", "shooting", "ID", "Symbol", excluded_ids=set(), n_clips=213)
    rows += portrait_rows()

    out_path = ROOT / "combined_labels.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["clip_id", "action", "camera_angle", "participant_id", "notes"])
        for clip_id, action, angle, note in rows:
            w.writerow([clip_id, action, angle, "", note])

    flagged = sum(1 for r in rows if r[3])
    print(f"wrote {len(rows)} rows -> {out_path}")
    print(f"flagged rows (missing/excluded/verify/portrait): {flagged}")


if __name__ == "__main__":
    main()
