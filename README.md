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


Distributed Processing with DuckDB Quack
Environment

Windows 11
Python 3.13.11
Conda 26.1.1
DuckDB 2.0.0
Database: photo_analysis.duckdb
Machines

winniesacerpc
mini
Connected using local Ethernet.

SSL is not required.

All workers should access the same photo folders using identical path names.

Starting the Quack Server
Store the database on winniesacerpc.

Example CMD window on winniesacerpc:

conda activate photos

set QUACK_TOKEN=mytoken

duckdb photo_analysis.duckdb
Inside DuckDB:

INSTALL quack;
LOAD quack;

CALL quack_start_server(
    host := '0.0.0.0',
    port := 9494,
    token := 'mytoken'
);
The host value 0.0.0.0 allows connections from any address on the local network.

Verify Network Connectivity
From mini:

ping winniesacerpc
and

telnet winniesacerpc 9494
(or another TCP test tool)

Worker 1
Run on winniesacerpc:

conda activate photos

set QUACK_TOKEN=mytoken

python analyze_photos_quack.py ^
    --quack quack:winniesacerpc:9494 ^
    --worker 1/2
Worker 2
Run on mini:

conda activate photos

set QUACK_TOKEN=mytoken

python analyze_photos_quack.py ^
    --quack quack:winniesacerpc:9494 ^
    --worker 2/2
Faster Caption Model
python analyze_photos_quack.py ^
    --caption-model Salesforce/blip-image-captioning-base
Resuming
The database stores:

path
file size
modification timestamp
Photos already analyzed are skipped automatically.

Workers may be stopped with Ctrl+C and restarted later.

Table
All results are stored in:

photos
within

photo_analysis.duckdb