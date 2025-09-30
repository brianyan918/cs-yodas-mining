import re
import json
from pathlib import Path
from multiprocessing import Pool, cpu_count
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

def process_lines(lines):
    results = []
    for line in lines:
        results.extend(segment_line(line))
    return results

def chunked_iterable(iterable, chunk_size):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= chunk_size:
            yield batch
            batch = []
    if batch:
        yield batch

def process_file_parallel(input_path, output_path, n_workers=None, chunk_size=1000):
    input_path = Path(input_path)
    output_path = Path(output_path)
    n_workers = n_workers or cpu_count()

    # Count total lines for progress bar
    with input_path.open("r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)

    # Read lines and batch
    with input_path.open("r", encoding="utf-8") as fin:
        batches = list(chunked_iterable(fin, chunk_size))

    all_segments = []
    with Pool(n_workers) as pool:
        for segments in tqdm(pool.imap(process_lines, batches), total=len(batches), desc="Processing"):
            all_segments.extend(segments)

    # Write output in order
    with output_path.open("w", encoding="utf-8") as fout:
        for seg in all_segments:
            fout.write(json.dumps(seg, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input text file")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel processes")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Number of lines per process batch")
    args = parser.parse_args()

    process_file_parallel(args.input, args.output, n_workers=args.workers, chunk_size=args.chunk_size)
