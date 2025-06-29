#!/usr/bin/env python3
import shutil
from pathlib import Path
import time

zarr_path = Path("data/RNA/processed/test/tabula_sapiens/all/chunk_0.zarr")
zip_path = zarr_path.with_suffix(".zarr.zip")

print(f"Starting compression of {zarr_path}...")
print(
    f"Original size: {sum(f.stat().st_size for f in zarr_path.rglob('*') if f.is_file()) / (1024 * 1024):.1f} MB"
)

start_time = time.time()

# Use shutil.make_archive for faster compression
shutil.make_archive(zip_path.with_suffix(""), "zip", zarr_path)

end_time = time.time()
print(f"Compression completed in {end_time - start_time:.2f} seconds")
print(f"Created: {zip_path}")

# Check the compressed size
if zip_path.exists():
    compressed_size = zip_path.stat().st_size / (1024 * 1024)
    print(f"Compressed size: {compressed_size:.1f} MB")
