# 2_evaluate_ipm_semantic.py
import json
import os
import re
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

class IPMEvaluator:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            "ipm_navigator_semantic",
            trust_remote_code=True,
            local_files_only=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            "models/phi-3-mini-4k-instruct",
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True,
            local_files_only=True
        )
        self.model = PeftModel.from_pretrained(base_model, "ipm_navigator_semantic")
        self.model.eval()

        with open("memory_store/memories.json", "r") as f:
            self.memories = json.load(f)
        with open("memory_store/train_data.json", "r") as f:
            self.train_data = json.load(f)

    def clean_pred(self, raw: str) -> str:
        raw = raw.strip().lower()
        raw = raw.replace("<|endoftext|>", "")
        parts = [p for p in re.split(r"[^a-z0-9]+", raw) if p]
        if len(parts) in (3, 4):
            candidate = "_".join(parts)
            if candidate in self.memories:
                return candidate;
            else:
                print(candidate);
                return "invalid";
            # return candidate if candidate in self.memories else "invalid"
        print(parts)
        return "invalid"

    def predict(self, question: str) -> str:
        prompt = (
    "You are a memory router. Given a user question, output ONLY the exact memory ID that contains the relevant fact.\n"
    "Memory IDs follow the format: domain_topic_entity (e.g., personal_pet_luna).\n"
    "Do NOT explain, do NOT add punctuation, do NOT generate new text.\n\n"
    f"Question: {question}\nMemory ID:"
)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=12,  # e.g., "personal_pet_luna" ~ 8-10 tokens
            temperature=0.0,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id
        )
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        pred_part = decoded.split("Memory ID:")[-1].strip()
        # print(f"Raw output: '{pred_part}'")  # ← 临时加这行
        raw_pred = pred_part.split()[0] if pred_part.split() else ""
        return self.clean_pred(raw_pred)

    def evaluate(self, num_samples=500):
        import random
        random.seed(42)
        eval_samples = random.sample(self.train_data, min(num_samples, len(self.train_data)))

        correct = 0
        total = 0
        for item in tqdm(eval_samples, desc="Evaluating IPM"):
            gold = item["memory_id"]
            if gold not in self.memories:
                continue
            pred = self.predict(item["question"])
            if pred == gold:
                correct += 1
            total += 1

        acc = correct / total if total > 0 else 0.0
        print(f"\n✅ IPM Top-1 Exact Recall: {acc:.1%} ({correct}/{total})")

        os.makedirs("results", exist_ok=True)
        with open("results/ipm_semantic_result.txt", "w") as f:
            f.write("=== IPM with Semantic Keys ===\n")
            f.write("Claim: 'A few tens of MB of LoRA can index millions of conversation turns.'\n")
            f.write(f"Memory count: {len(self.memories)}\n")
            f.write(f"LoRA size: ~180 MB (r=256)\n")
            f.write(f"Semantic ID format: domain_topic_entity\n")
            f.write(f"Top-1 Exact Recall: {acc:.1%}\n")
            if acc >= 0.90:
                f.write("\n✅ SUCCESS: Semantic IPM works!\n")
        print("Result saved to results/ipm_semantic_result.txt")

if __name__ == "__main__":
    evaluator = IPMEvaluator()
    evaluator.evaluate()