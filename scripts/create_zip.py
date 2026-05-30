# scripts/create_zip.py
import zipfile
import os
import json

ZIP_PATH = "facetbench_deliverable.zip"

with zipfile.ZipFile(ZIP_PATH, 'w', zipfile.ZIP_DEFLATED) as zf:
    # Add all raw conversations
    for fname in sorted(os.listdir("data/conversations/raw")):
        if fname.endswith(".json"):
            zf.write(
                os.path.join("data/conversations/raw", fname),
                os.path.join("conversations/raw", fname)
            )

    # Add all scored conversations
    for fname in sorted(os.listdir("data/conversations/scored")):
        if fname.endswith(".json"):
            zf.write(
                os.path.join("data/conversations/scored", fname),
                os.path.join("conversations/scored", fname)
            )

    # Add facets
    zf.write("data/facets/facets.json", "facets/facets.json")

print(f"ZIP created: {ZIP_PATH}")

# Verify counts
with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
    names = zf.namelist()
    raw = [n for n in names if "raw/" in n]
    scored = [n for n in names if "scored/" in n]
    print(f"Raw conversations: {len(raw)}")
    print(f"Scored conversations: {len(scored)}")
    print(f"Total files: {len(names)}")