#!/usr/bin/env python3
import os
import json

# Path to the FLEURS data directory
DATA_DIR = "../corpora/fleurs/data"
OUTPUT_FILE = "data/fleurs_test.jsonl"

# Open output JSONL file
with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
    # Loop over all language pair directories
    for lang_pair in os.listdir(DATA_DIR):
        lang_dir = os.path.join(DATA_DIR, lang_pair)
        if not os.path.isdir(lang_dir):
            continue  # skip files like zips

        tsv_file = os.path.join(lang_dir, "test.tsv")
        if not os.path.exists(tsv_file):
            continue  # skip if no test.tsv

        with open(tsv_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) != 7:
                    print(f"Skipping malformed line in {tsv_file}: {line}")
                    continue

                id_, filename, text, text_char, duration, gender = parts[0], parts[1], parts[2], parts[3], parts[5], parts[6]

                entry = {
                    "id": id_,
                    "file_name": os.path.join(lang_dir, filename),
                    "text": text,
                    "text_char": text_char,
                    "duration": float(duration) / 1000.0,  # assuming duration is in milliseconds
                    "gender": gender,
                    "language_pair": lang_pair
                }

                out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"Done! JSONL written to {OUTPUT_FILE}")
