import json
import argparse

def clean_jsonl(input_file, output_file, default_value={"text": ""}):
    with open(input_file, "r", encoding="utf-8") as fin, \
         open(output_file, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                obj = default_value
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True, help="Path to input JSONL file")
    parser.add_argument("--output_file", required=True, help="Path to output cleaned JSONL file")
    args = parser.parse_args()

    clean_jsonl(args.input_file, args.output_file)
