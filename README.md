analyze_photos.py - caption, keyword and quality-score JPG photos into DuckDB.

Everything runs locally (no photo leaves the PC):
  caption   BLIP   (Salesforce/blip-image-captioning-large)
  keywords  CLIP   (openai/clip-vit-base-patch32), zero-shot against KEYWORDS below
  quality   Pillow + NumPy (sharpness, brightness, contrast, clipping)

Setup (once):
  conda create -n photos python=3.13 -y
  conda activate photos
  pip install duckdb pillow numpy torch transformers

Run:
  python analyze_photos.py
  python analyze_photos.py --limit 20            # quick trial
  python analyze_photos.py "D:\\other\\folder"     # different folders
# Use this model for faster processing with less accuracy
  python analyze_photos.py --caption-model   Salesforce/blip-image-captioning-base

The run is resumable: photos already in the database (same size and
modified time) are skipped, so you can stop with Ctrl+C and restart.
