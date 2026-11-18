# analyze_photos_quack.py
"""# co_analyze_photos_quack.py
# V2
#
# Optimized for:
#   DuckDB 2.x + Quack
#   Python 3.13
#   Windows 11
#   External USB3 photo drives
#
# Changes from V1:
#   - BLIP Large -> BLIP Base
#   - Default batch size 32
#
# Expected improvements:
#   - substantially lower VRAM usage
#   - 2x-4x faster caption generation
#   - improved GPU utilization
#   - unchanged database schema and resumability
"""

import argparse
import os
import sys
import time
import warnings
import zlib
from datetime import datetime
from pathlib import Path

import duckdb
import numpy as np
from PIL import Image, ImageOps

DEFAULT_ROOTS       = [r"H:\Bilder", r"O:\bilder\1-d7100"]
DEFAULT_DB          = "photo_analysis.duckdb"
CAPTION_MODEL       = "Salesforce/blip-image-captioning-base"
ANALYSIS_EDGE       = 768
DEFAULT_BATCH_SIZE  = 32

COMMIT_EVERY = 400
KEYWORD_MODEL = "openai/clip-vit-base-patch32"
COMMIT_EVERY = 400  # photos per database commit

# Vocabulary for keywords. Edit freely; delete the database rows (or the
# file) afterwards if you want existing photos re-tagged.
KEYWORDS = [
    "landscape", "mountains", "forest", "trees", "lake", "river", "waterfall",
    "sea", "beach", "coast", "sky", "clouds", "sunset", "sunrise", "night sky",
    "stars", "snow", "fog", "rain", "desert", "field", "meadow", "flowers",
    "garden", "park", "city", "street", "skyline", "building", "architecture",
    "church", "castle", "bridge", "ruins", "monument", "interior", "museum",
    "portrait", "person", "group of people", "child", "family", "wedding",
    "party", "concert", "sports", "hiking", "cycling", "skiing", "swimming",
    "dog", "cat", "bird", "horse", "insect", "butterfly", "wildlife", "fish",
    "food", "drink", "car", "train", "airplane", "boat", "bicycle",
    "macro close-up", "panorama", "black and white", "silhouette",
    "reflection", "fireworks", "lights at night", "market", "harbor",
    "village", "road", "path", "rocks", "cave", "autumn leaves", "winter",
    "document or text", "screenshot", "still life", "art", "statue",
]
MAX_KEYWORDS = 6
MIN_KEYWORD_SCORE = 0.04

SCHEMA = """
CREATE TABLE IF NOT EXISTS {table} (
    path                   VARCHAR PRIMARY KEY,
    root                   VARCHAR,
    file_name              VARCHAR,
    file_size              BIGINT,
    file_mtime             TIMESTAMP,
    width                  INTEGER,
    height                 INTEGER,
    megapixels             DOUBLE,
    sharpness              DOUBLE,   -- variance of Laplacian; higher = sharper
    brightness             DOUBLE,   -- mean luminance, 0-255
    contrast               DOUBLE,   -- std dev of luminance
    shadows_clipped_pct    DOUBLE,   -- % of pixels that are near black
    highlights_clipped_pct DOUBLE,   -- % of pixels that are near white
    caption                VARCHAR,
    keywords               VARCHAR[],
    keyword_scores         DOUBLE[],
    caption_model          VARCHAR,
    keyword_model          VARCHAR,
    analyzed_at            TIMESTAMP,
    error                  VARCHAR
)
"""


def find_photos(roots):
    """Yield (root, path) for every .jpg/.JPG under the given folders."""
    for root in roots:
        if not os.path.isdir(root):
            print(f"WARNING: folder not reachable, skipped: {root}", file=sys.stderr)
            continue
        for dirpath, _dirs, files in os.walk(root):
            for name in files:
                if name.lower().endswith(".jpg"):
                    yield root, os.path.join(dirpath, name)


def load_image(path):
    """Return (analysis-size RGB image, original width, original height)."""
    img = Image.open(path)
    width, height = img.size
    img.draft("RGB", (ANALYSIS_EDGE, ANALYSIS_EDGE))  # fast reduced-size JPEG decode
    orientation = img.getexif().get(0x0112, 1)
    img = ImageOps.exif_transpose(img).convert("RGB")
    img.thumbnail((ANALYSIS_EDGE, ANALYSIS_EDGE), Image.LANCZOS)
    if orientation in (5, 6, 7, 8):  # stored rotated by 90 degrees
        width, height = height, width
    return img, width, height


def quality_metrics(img):
    gray = np.asarray(img.convert("L"), dtype=np.float32)
    lap = (gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:]
           - 4.0 * gray[1:-1, 1:-1])
    return {
        "sharpness": float(lap.var()),
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "shadows_clipped_pct": float((gray <= 4).mean() * 100),
        "highlights_clipped_pct": float((gray >= 251).mean() * 100),
    }


class Analyzer:
    """Local caption (BLIP) and keyword (CLIP) models."""

    def __init__(self, caption_model, keyword_model):
        import torch
        from transformers import (BlipForConditionalGeneration, BlipProcessor,
                                  CLIPModel, CLIPProcessor)

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading models on {self.device} (first run downloads about 2.5 GB) ...")
        self.cap_proc = BlipProcessor.from_pretrained(caption_model)
        self.cap_model = BlipForConditionalGeneration.from_pretrained(caption_model).to(self.device).eval()
        self.kw_proc = CLIPProcessor.from_pretrained(keyword_model)
        self.kw_model = CLIPModel.from_pretrained(keyword_model).to(self.device).eval()
        self.kw_text = self.kw_proc(text=[f"a photo of {k}" for k in KEYWORDS],
                                    return_tensors="pt", padding=True).to(self.device)

    def analyze(self, images):
        """Return a list of (caption, keywords, scores), one per image."""
        torch = self.torch
        with torch.inference_mode():
            cap_in = self.cap_proc(images=images, return_tensors="pt").to(self.device)
            ids = self.cap_model.generate(**cap_in, max_new_tokens=40, num_beams=3)
            captions = [c.strip() for c in self.cap_proc.batch_decode(ids, skip_special_tokens=True)]

            pixels = self.kw_proc(images=images, return_tensors="pt")["pixel_values"].to(self.device)
            probs = self.kw_model(pixel_values=pixels, **self.kw_text).logits_per_image.softmax(dim=-1).cpu().numpy()

        results = []
        for caption, row in zip(captions, probs):
            top = [i for i in np.argsort(row)[::-1][:MAX_KEYWORDS] if row[i] >= MIN_KEYWORD_SCORE]
            results.append((caption, [KEYWORDS[i] for i in top], [round(float(row[i]), 4) for i in top]))
        return results


COLUMNS = ["path", "root", "file_name", "file_size", "file_mtime", "width", "height",
           "megapixels", "sharpness", "brightness", "contrast", "shadows_clipped_pct",
           "highlights_clipped_pct", "caption", "keywords", "keyword_scores",
           "caption_model", "keyword_model", "analyzed_at", "error"]
INSERT = (f"INSERT OR REPLACE INTO {{table}} ({', '.join(COLUMNS)}) "
          f"VALUES ({', '.join('?' * len(COLUMNS))})")


def connect(args):
    """Open the local database file, or attach a Quack server.

    Returns (connection, table name).
    """
    if not args.quack:
        return duckdb.connect(args.db), "photos"
    token = args.token or os.environ.get("QUACK_TOKEN")
    if not token:
        sys.exit("Quack needs a token: use --token or set QUACK_TOKEN.")
    con = duckdb.connect(":memory:")
    con.execute("INSTALL quack")
    con.execute("LOAD quack")
    quote = lambda v: v.replace("'", "''")
    con.execute(f"ATTACH '{quote(args.quack)}' AS photodb "
                f"(TOKEN '{quote(token)}', DISABLE_SSL {'false' if args.ssl else 'true'})")
    return con, "photodb.photos"


def parse_worker(value):
    """'2/3' -> (index 1, count 3)."""
    try:
        i, n = (int(x) for x in value.split("/"))
        assert 1 <= i <= n
    except Exception:
        raise argparse.ArgumentTypeError("use the form I/N, e.g. 1/2")
    return i - 1, n


def is_mine(path, worker):
    """Split photos between workers by a hash of the path."""
    index, count = worker
    return zlib.crc32(path.lower().encode("utf-8")) % count == index


def process_batch(con, table, analyzer, batch, args):
    """batch: list of (root, path, size, mtime). Writes one row per photo.

    Returns (errors, excluded). Photos over Pillow's pixel limit
    (178,956,970 pixels) are excluded: nothing is written for them.
    """
    rows, images, ok, excluded = [], [], [], 0
    for root, path, size, mtime in batch:
        row = dict.fromkeys(COLUMNS)
        row.update(path=path, root=root, file_name=os.path.basename(path), file_size=size,
                   file_mtime=mtime, caption_model=args.caption_model,
                   keyword_model=args.keyword_model)
        try:
            img, w, h = load_image(path)
            row.update(width=w, height=h, megapixels=round(w * h / 1e6, 2), **quality_metrics(img))
            images.append(img)
            ok.append(row)
        except Image.DecompressionBombError:
            excluded += 1
            continue
        except Exception as exc:  # corrupt or unreadable file
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)

    if images:
        try:
            for row, (caption, kws, scores) in zip(ok, analyzer.analyze(images)):
                row.update(caption=caption, keywords=kws, keyword_scores=scores)
        except Exception as exc:
            for row in ok:
                row["error"] = f"{type(exc).__name__}: {exc}"

    now = datetime.now()
    for row in rows:
        row["analyzed_at"] = now
    if rows:  # empty when every photo in the batch was excluded as too large
        con.executemany(INSERT.format(table=table), [[row[c] for c in COLUMNS] for row in rows])
    return sum(1 for r in rows if r["error"]), excluded


def main():
    ap = argparse.ArgumentParser(description="Analyze JPG photos into a DuckDB database.")
    ap.add_argument("roots", nargs="*", default=DEFAULT_ROOTS, help="folders to scan recursively")
    ap.add_argument("--db", default=DEFAULT_DB, help=f"database file (default {DEFAULT_DB})")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument(
        "--batch-size",
        type=int,
        default=32,
    help="number of photos processed by BLIP/CLIP in one batch"
)
    ap.add_argument("--commit-every", type=int, default=COMMIT_EVERY,
                    help=f"commit after this many photos (default {COMMIT_EVERY})")
    ap.add_argument("--quack", metavar="URI",
                    help="Quack server to use instead of --db, e.g. quack:192.168.1.20:9494")
    ap.add_argument("--token", help="Quack token (default: environment variable QUACK_TOKEN)")
    ap.add_argument("--ssl", action="store_true", help="connect to the Quack server over HTTPS")
    ap.add_argument("--worker", type=parse_worker, default=(0, 1), metavar="I/N",
                    help="this is worker I of N; each worker takes its own share of the photos")
    ap.add_argument("--limit", type=int, help="analyze at most this many new photos")
    ap.add_argument("--caption-model", default=CAPTION_MODEL)
    ap.add_argument("--keyword-model", default=KEYWORD_MODEL)
    args = ap.parse_args()

    con, table = connect(args)
    con.execute(SCHEMA.format(table=table))
    con.execute(f"DELETE FROM {table} WHERE error LIKE 'DecompressionBombError%'")
    warnings.simplefilter("ignore", Image.DecompressionBombWarning)
    done = {p: (s, m) for p, s, m in con.execute(
        f"SELECT path, file_size, file_mtime FROM {table} WHERE error IS NULL").fetchall()}

    print("Scanning folders ...")
    todo = []
    found = 0
    for root, path in find_photos(args.roots):
        if not is_mine(path, args.worker):
            continue
        found += 1
        st = os.stat(path)
        mtime = datetime.fromtimestamp(st.st_mtime)
        if done.get(path) != (st.st_size, mtime):
            todo.append((root, path, st.st_size, mtime))
    pending = len(todo)
    if args.limit:
        todo = todo[:args.limit]
    if args.worker[1] > 1:
        print(f"Worker {args.worker[0] + 1} of {args.worker[1]}: counts below are this worker's share.")
    print(f"{found} photos found, {found - pending} already done, {len(todo)} to analyze now.")
    if not todo:
        return

    analyzer = Analyzer(args.caption_model, args.keyword_model)
    start, errors, excluded, uncommitted = time.time(), 0, 0, 0
    con.execute("BEGIN")
    try:
        for i in range(0, len(todo), args.batch_size):
            batch = todo[i:i + args.batch_size]
            err, exc = process_batch(con, table, analyzer, batch, args)
            errors += err
            excluded += exc
            uncommitted += len(batch)
            if uncommitted >= args.commit_every:
                con.execute("COMMIT")
                con.execute("BEGIN")
                uncommitted = 0
            n = min(i + args.batch_size, len(todo))
            rate = n / (time.time() - start)
            eta_min = (len(todo) - n) / rate / 60
            print(f"\r{n}/{len(todo)}  {rate:.2f} photos/s  ETA {eta_min:.0f} min  errors {errors}  "
                  f"too large {excluded}",
                  end="", flush=True)
    except KeyboardInterrupt:
        print("\nStopped - run again to continue where it left off.")
    finally:
        print()
        con.execute("COMMIT")
        con.close()
    print(f"Done. Results are in {args.quack or Path(args.db).resolve()}")


if __name__ == "__main__":
    main()