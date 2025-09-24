#!/usr/bin/env python3
import argparse
import json
from vllm import LLM, SamplingParams

def main():
    parser = argparse.ArgumentParser(description="Language Identification using vLLM")
    parser.add_argument("--model", type=str, default="qwen3-4b-Instruct", help="vLLM model to use")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file with 'text' field")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file with predicted language")
    parser.add_argument("--prompt_template", type=str,
                        default="Identify the language of the following text. Respond with a single language code or name:\n{text}",
                        help="Prompt template. Use {text} as placeholder for input text.")
    parser.add_argument("--max_tokens", type=int, default=20, help="Maximum tokens to generate")
    args = parser.parse_args()

    # Initialize the LLM
    llm = LLM(args.model)

    with open(args.input, "r", encoding="utf-8") as f_in, \
         open(args.output, "w", encoding="utf-8") as f_out:

        for line in f_in:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            text = data.get("text", "")
            if not text:
                data["language_pred"] = None
                f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
                continue

            # Prepare prompt
            prompt = args.prompt_template.format(text=text)

            # Generate output
            sampling_params = SamplingParams(max_output_tokens=args.max_tokens)
            result = llm.generate([prompt], sampling_params=sampling_params)
            # vLLM returns a generator; grab the first output
            pred_text = next(result).outputs[0].text.strip()

            # Save prediction
            data["language_pred"] = pred_text
            f_out.write(json.dumps(data, ensure_ascii=False) + "\n")

    print(f"Predictions written to {args.output}")

if __name__ == "__main__":
    main()
