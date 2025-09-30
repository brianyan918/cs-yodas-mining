import re
import json
from pathlib import Path
from tqdm import tqdm

HEADER_RE = re.compile(r'^(\S+)\s*<(\w+)><(\w+)>')
SEGMENT_RE = re.compile(r'(.*?)<(\d+\.\d+)><(\d+\.\d+)>')  # non-greedy

def segment_line(line):
    line = line.strip()
    if not line:
        return []

    header_match = HEADER_RE.match(line)
    if not header_match:
        print(f"Skipping line (cannot parse uttid/lang/task): {line[:60]}...")
        return []

    uttid = header_match.group(1)
    language = header_match.group(2)
    rest = line[header_match.end():].strip()

    segments = []
    first_segment = True
    for match in SEGMENT_RE.finditer(rest):
        text = match.group(1).strip()
        start = float(match.group(2))
        end = float(match.group(3))

        if first_segment and text.startswith("<0.00>"):
            text = text[len("<0.00>"):].strip()
        first_segment = False

        start_id = f"{int(round(start*100)):03d}"
        end_id = f"{int(round(end*100)):03d}"
        segment_id = f"{uttid}_{start_id}_{end_id}"

        segments.append({
            "id": segment_id,
            "text": text,
            "language": language
        })
    return segments

def process_file(input_path, output_path):
    input_path = Path(input_path)
    output_path = Path(output_path)

    # Count lines once for progress bar
    with input_path.open("r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)

    with input_path.open("r", encoding="utf-8") as fin, \
         output_path.open("w", encoding="utf-8") as fout:
        for line in tqdm(fin, total=total_lines, desc="Processing"):
            segs = segment_line(line)
            for seg in segs:
                fout.write(json.dumps(seg, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input text file")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    args = parser.parse_args()

    process_file(args.input, args.output)
