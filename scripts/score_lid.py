import json
import ast
import argparse
from collections import Counter, defaultdict

# Mapping to ISO-639-3 (extend as needed)
LANG_MAP = {
    "English": "eng",
    "Arabic": "ara",
    "Mandarin": "cmn",
    "Chinese": "cmn",
    "Spanish": "spa",
    "French": "fra",
    # add more if needed
}

def parse_hyp(text):
    """Parse hyp string into normalized concatenated label."""
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        try:
            obj = ast.literal_eval(text)
        except Exception:
            return ""
    langs = obj.get("languages", [])
    mapped = [LANG_MAP.get(lang, lang[:3].lower()) for lang in langs]
    return "-".join(sorted(mapped))  # enforce canonical order

def main(hyps_file, refs_file):
    hyps, refs = [], []

    with open(hyps_file, "r") as hf, open(refs_file, "r") as rf:
        for h, r in zip(hf, rf):
            hyps.append(parse_hyp(h.strip()))
            refs.append(json.loads(r.strip())["language"].lower())

    # Overall accuracy
    overall_acc = sum(h == r for h, r in zip(hyps, refs)) / len(refs)

    # Code-switched precision (hyps with dash)
    cs_hyp_indices = [i for i, h in enumerate(hyps) if "-" in h]
    cs_correct = sum(hyps[i] == refs[i] for i in cs_hyp_indices)
    cs_precision = cs_correct / len(cs_hyp_indices) if cs_hyp_indices else 0.0

    # Code-switched recall (refs with dash)
    cs_ref_indices = [i for i, r in enumerate(refs) if "-" in r]
    cs_recall = sum(hyps[i] == refs[i] for i in cs_ref_indices) / len(cs_ref_indices) if cs_ref_indices else 0.0

    # Accuracy by class
    class_acc = defaultdict(lambda: [0, 0])  # [correct, total]
    for h, r in zip(hyps, refs):
        class_acc[r][1] += 1
        if h == r:
            class_acc[r][0] += 1
    class_acc = {cls: correct/total for cls, (correct, total) in class_acc.items()}

    # Confusion pairs
    confusions = Counter((r, h) for r, h in zip(refs, hyps) if r != h)

    # Code-switched confusion pairs (only where hyp has a dash)
    cs_confusions = Counter((r, h) for r, h in zip(refs, hyps) if r != h and "-" in h)

    # Print results
    print(f"Overall Accuracy: {overall_acc*100:.2f}%")
    print(f"Code-switched Precision: {cs_precision*100:.2f}%")
    print(f"Code-switched Recall: {cs_recall*100:.2f}%")
    print("\nAccuracy by class:")
    for cls, acc in class_acc.items():
        print(f"  {cls}: {acc*100:.2f}%")

    print("\nTop overall confusion pairs:")
    for (r, h), count in confusions.most_common(10):
        print(f"  {r} → {h}: {count}")

    print("\nTop code-switched confusion pairs (hyp has '-'):")    
    for (r, h), count in cs_confusions.most_common(10):
        print(f"  {r} → {h}: {count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hyps", required=True, help="File with hypothesis JSON lines")
    parser.add_argument("--refs", required=True, help="File with reference labels")
    args = parser.parse_args()
    main(args.hyps, args.refs)
