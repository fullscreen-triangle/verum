"""
Inference with the Verum-specialized LLM.

Usage:
    python inference.py --model verum-model --prompt "What is the universal transport formula?"
"""
import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="verum-model", help="Model path")
    parser.add_argument("--prompt", required=True, help="Input prompt")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max output tokens")
    args = parser.parse_args()

    model_path = os.path.join(os.path.dirname(__file__), args.model)

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel  # noqa: F401
    except ImportError:
        print("Install: pip install transformers peft torch")
        return

    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Run train.py first.")
        return

    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto"
    )

    inputs = tokenizer(args.prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nPrompt: {args.prompt}")
    print(f"\nResponse: {response[len(args.prompt):]}")


if __name__ == "__main__":
    main()
