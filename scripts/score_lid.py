import json
import ast
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Mapping to ISO-639-3 (extend as needed)
LANG_MAP = {
    "English": "Eng",
    "Arabic": "Ara",
    "Mandarin": "Cmn",
    "Chinese": "Cmn",
    "Spanish": "Spa",
    "French": "Fra",
    # add more if needed
}

def parse_hyp(text):
    """Parse hyp string into normalized concatenated label."""
    # Handle invalid JSON (single quotes)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        obj = ast.literal_eval(text)

    try:
        langs = obj["languages"]
    except:
        obj = json.loads(obj)
        langs = obj["languages"]

    mapped = [LANG_MAP.get(lang, lang[:3].title()) for lang in langs]
    return "-".join(sorted(mapped))  # enforce canonical order

def main(hyps_file, refs_file):
    hyps, refs = [], []

    with open(hyps_file, "r") as hf, open(refs_file, "r") as rf:
        for h, r in zip(hf, rf):
            hyps.append(parse_hyp(h.strip()))
            refs.append(r.strip())

    # Metrics
    accuracy = accuracy_score(refs, hyps)
    precision = precision_score(refs, hyps, average="macro", zero_division=0)
    recall = recall_score(refs, hyps, average="macro", zero_division=0)
    f1 = f1_score(refs, hyps, average="macro", zero_division=0)

    print("Accuracy:", accuracy * 100)
    print("Precision:", precision * 100)
    print("Recall:", recall * 100)
    print("F1:", f1 * 100)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hyps", required=True, help="File with hypothesis JSON lines")
    parser.add_argument("--refs", required=True, help="File with reference labels")
    args = parser.parse_args()

    main(args.hyps, args.refs)
