"""Import a Label Studio "YOLO" export (zip with images/, labels/, classes.txt) into a training dataset.

  python3 detection/import_label_studio.py "labalestudio yolo files /project-1-at-2026-11-27-09-33-68da4995.zip"
  -> detection/data/label_studio/dataset/{images,labels}/{train,val}   (classes 0 egg, 1 ball)

  - Classes are mapped by NAME from the export's classes.txt (Label Studio sorts them, e.g. 0 ball / 1 egg) to this
    project's order: 0 egg, 1 ball. Unknown names are reported and skipped.
  - Only the plain YOLO export works (class cx cy w h); a YOLO-OBB export (8 corner numbers) is rejected - export
    "YOLO" (with images) in Label Studio.
  - Train / val split by frame number in blocks (neighbouring frames of a video look alike, so they stay together):
    frames in every 5th block of 150 go to val.
  - Several exports can be imported one after another; files are named <recording>_<frame>.jpg, so re-importing the
    same image replaces it.
"""
import argparse
import re
import shutil
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUR_CLASSES = {"egg": 0, "ball": 1}
ap = argparse.ArgumentParser()
ap.add_argument("zips", nargs="+", help="Label Studio YOLO export(s) (.zip or an unpacked folder)")
ap.add_argument("--out", default=str(ROOT / "detection" / "data" / "label_studio" / "dataset"))
ap.add_argument("--block", type=int, default=150, help="frames per split block")
args = ap.parse_args()

out = Path(args.out)
for s in ("train", "val"):
    (out / "images" / s).mkdir(parents=True, exist_ok=True)
    (out / "labels" / s).mkdir(parents=True, exist_ok=True)

total = {"train": 0, "val": 0, "egg": 0, "ball": 0, "skipped": 0}
for z in args.zips:
    z = Path(z)
    with tempfile.TemporaryDirectory() as tmp:
        src = z
        if z.suffix == ".zip":
            with zipfile.ZipFile(z) as f:
                f.extractall(tmp)
            src = Path(tmp)
        names = [l.strip() for l in (src / "classes.txt").read_text().splitlines() if l.strip()]
        remap = {i: OUR_CLASSES.get(n) for i, n in enumerate(names)}
        unknown = [n for n in names if n not in OUR_CLASSES]
        print(f"{z.name}: classes {names} -> {{{', '.join(f'{n}: {OUR_CLASSES.get(n)}' for n in names)}}}"
              + (f"  (unknown, skipped: {unknown})" if unknown else ""))
        imgs = {p.stem: p for p in (src / "images").glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")}
        if not imgs:
            raise SystemExit(f"{z}: no images/ - export 'YOLO' WITH images in Label Studio")
        for lab in sorted((src / "labels").glob("*.txt")):
            if lab.stem not in imgs:
                total["skipped"] += 1
                continue
            rows = []
            for line in lab.read_text().splitlines():
                v = line.split()
                if not v:
                    continue
                if len(v) != 5:
                    raise SystemExit(f"{lab.name}: {len(v)} numbers per box - this is not the plain YOLO export "
                                     "(YOLO-OBB?). Export 'YOLO' in Label Studio.")
                c = remap.get(int(v[0]))
                if c is None:
                    continue
                rows.append(" ".join([str(c), *v[1:]]))
                total["egg" if c == 0 else "ball"] += 1
            # "00a256b4-rec_20261127_094904_000580" (or "22028c83-ac445f87-rec_..." when already exported images are
            # uploaded again) -> recording rec_20261127_094904, frame 580
            name = re.sub(r"^([0-9a-f]{8}-)+", "", lab.stem)
            m = re.search(r"_(\d+)$", name)
            frame = int(m.group(1)) if m else hash(name) % 10000
            split = "val" if (frame // args.block) % 5 == 4 else "train"
            old = out / "labels" / split / f"{name}.txt"
            if old.exists() and sorted(r.split()[0] for r in old.read_text().splitlines() if r.strip()) != sorted(r.split()[0] for r in rows):
                print(f"  WARNING {name}: labels differ from the earlier import (classes "
                      f"{sorted(r.split()[0] for r in old.read_text().splitlines() if r.strip())} -> {sorted(r.split()[0] for r in rows)}),"
                      " replaced - did you label every egg AND ball?")
            shutil.copy2(imgs[lab.stem], out / "images" / split / f"{name}{imgs[lab.stem].suffix.lower()}")
            (out / "labels" / split / f"{name}.txt").write_text("\n".join(rows) + ("\n" if rows else ""))
            total[split] += 1

print(f"imported: train {total['train']}, val {total['val']} images; {total['egg']} egg + {total['ball']} ball boxes"
      + (f"; {total['skipped']} labels without image skipped" if total["skipped"] else "") + f" -> {out}")
