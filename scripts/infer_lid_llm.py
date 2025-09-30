#!/usr/bin/env python3
import argparse
import json
from vllm import LLM, SamplingParams
from tqdm import tqdm
from transformers import AutoTokenizer

ic_egs = [["这些 地区 人口 稀少 ， often 不 存在 光 污染 的 问题 ， 你 also 能 欣赏 到 璀 璨 星空 。",["Chinese", "English"]],\
        ["这些地区人口稀少，往往不存在光污染的问题，你也能欣赏到璀璨星空", ["Chinese"]],
        ["These areas are sparsely populated, light pollution is often not a problem, and you can also enjoy the brilliant starry sky.", ["English"]]]

def construct_prompt(text, tokenizer, think):
    messages = [
        {"role": "system", "content": "You are performing text-based language identification. We are trying to identify code-mixed or code-switched utterances."},
    ]
    prompt = """Text: ```{text}```

For the given text in triple backticks identify ALL languages that appear. There may be only a single language or multiple languages that are code-mixed together. Your final answer should list the languages in order of prevalance.\nCode-mixing, or code-switching, is defined as the alternation of two languages within a single discourse, sentence, or constituent. Double check whether the text contains code-switching by reviewing word-by-word. Do not simply glance at the overall sentence and only write down the dominant language.\nFormat your response as a json object.
    """
    for egs in ic_egs:
        messages.append({"role": "user", "content": prompt.format(text=egs[0])})
        response = {"languages": egs[1]}
        messages.append({"role": "assistant", "content": str(response)})

    messages.append({"role": "user", "content": prompt.format(text=text)})
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=think
    )
    # Compute token length
    tokenized = tokenizer.tokenize(prompt_text)
    token_length = len(tokenized)

    return prompt_text, token_length

def validate(text):
    # must pass json
    try:
        parsed = json.loads(text.replace("'", '"'))
        return True
    except json.JSONDecodeError:
        # retry with "fix JSON" prompt
        return False

def fallback(llm, prompt, sampling_params):
    results = llm.generate([prompt], sampling_params=sampling_params)
    for result in results[0].outputs:
        if validate(result.text):
            return result.text.replace("\n", "\\n").strip()
    return ""

def main():
    parser = argparse.ArgumentParser(description="Language Identification using vLLM")
    parser.add_argument("--model", type=str, default="qwen3-4b-Instruct", help="vLLM model to use")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file with 'text' field")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file with predicted language")
    parser.add_argument("--max_tokens", type=int, default=500, help="Maximum tokens to generate")
    parser.add_argument("--max_model_len", type=int, default=5000, help="Maximum model length")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--think", type=int, default=0, help="Reasoning mode")
    args = parser.parse_args()

    # Initialize the LLM
    llm = LLM(args.model, max_model_len=args.max_model_len)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    sampling_params = SamplingParams(max_tokens=args.max_tokens, temperature=0.7, top_p=0.95, top_k=20, min_p=0)
    fallback_sampling_params = SamplingParams(max_tokens=args.max_tokens, n=10, temperature=0.7, top_p=0.95, top_k=20, min_p=0)

    input_data = open(args.input, "r").readlines()

    batch = []

    with open(args.output, "w", encoding="utf-8") as f_out, open(args.output+".prompt", "w", encoding="utf-8") as f_out_p:
        pass
    for i, line in tqdm(enumerate(input_data)):
        with open(args.output, "a", encoding="utf-8") as f_out, open(args.output+".prompt", "a", encoding="utf-8") as f_out_p:
            data = json.loads(line.strip())
            if "text" not in data:
                data["language_pred"] = None
                f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
                continue
            else:
                text = data.get("text", "").replace("**", "")

            # Prepare prompt
            prompt, prompt_len = construct_prompt(text, tokenizer, think=(args.think==1))
            if prompt_len > args.max_model_len - args.max_tokens:
                prompt = r"This is a dummy prompt. Output an json object: {}"
            batch.append(prompt)

            # Generate output
            if len(batch) >= args.batch_size or i == len(input_data) - 1:
                results = llm.generate(batch, sampling_params=sampling_params)
                for p, r in zip(batch, results):
                    prompt_text = p.replace("\n", "\\n").strip()
                    pred_text = r.outputs[0].text.replace("\n", "\\n").strip()
                    if not validate(pred_text):
                        # fallback
                        pred_text = fallback(llm, p, fallback_sampling_params)

                    # Save prediction
                    f_out.write(pred_text + "\n")
                    f_out_p.write(prompt_text + "\n")
                f_out.flush()
                f_out_p.flush()
                batch = []

if __name__ == "__main__":
    main()
