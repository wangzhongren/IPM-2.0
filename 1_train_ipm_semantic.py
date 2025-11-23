# 1_train_ipm_semantic.py
import os
import json
import random
from datetime import datetime, timedelta
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset
import torch

def generate_ipm_semantic_data():
    """
    Generate 100 memories with structured semantic keys:
    <domain>_<topic>_<entity>
    """
    domains = ["personal", "work", "opinion", "plan", "preference"]
    topics = {
        "personal": ["pet", "gear"],
        "work": ["project", "meeting"],
        "opinion": ["llm"],
        "plan": ["travel"],
        "preference": ["food", "language"]
    }
    entities = {
        "pet": ["luna", "max"],
        "gear": ["herman_miller", "steelcase"],
        "project": ["alpha", "beta"],
        "meeting": ["standup"],
        "llm": ["grok4", "claude4"],
        "travel": ["japan", "iceland"],
        "food": ["pizza", "sushi"],
        "language": ["python", "rust"]
    }

    memories = {}
    train_data = []
    count = 0

    for domain in domains:
        for topic in topics[domain]:
            for entity in entities[topic]:
                if count >= 100:
                    break

                # Construct semantic ID
                mem_id = f"{domain}_{topic}_{entity}"  # e.g., personal_pet_luna

                # Generate memory text
                if topic == "pet":
                    text = f"My {entity}'s name is Luna." if entity == "luna" else f"My dog's name is Max."
                    base_q = f"What is my {entity} called?"
                elif topic == "gear":
                    chair = "Herman Miller Aeron" if "herman" in entity else "Steelcase Leap"
                    text = f"My office chair is a {chair}."
                    base_q = "What chair do I have?"
                elif topic == "project":
                    date = (datetime(2025, 1, 1) + timedelta(days=random.randint(30, 365))).strftime("%Y-%m-%d")
                    text = f"The deadline for Project {entity.capitalize()} is {date}."
                    base_q = f"When is Project {entity.capitalize()} due?"
                elif topic == "meeting":
                    text = "I really hate long daily stand-up meetings."
                    base_q = "What do I hate about work?"
                elif topic == "llm":
                    model = "Grok-4" if entity == "grok4" else "Claude 4"
                    text = f"I think {model} is the strongest LLM right now."
                    base_q = f"Which model do I think is best?"
                elif topic == "travel":
                    year = random.choice([2026, 2027])
                    text = f"I'm planning a trip to {entity.capitalize()} in {year}."
                    base_q = f"Where am I traveling in {year}?"
                elif topic == "food":
                    rest = "Lou Malnati's" if entity == "pizza" else "Sukiyabashi Jiro"
                    text = f"I love {entity} from {rest}."
                    base_q = f"What food do I love?"
                elif topic == "language":
                    lang = "Python" if entity == "python" else "Rust"
                    text = f"My favorite programming language is {lang}."
                    base_q = "Favorite programming language?"

                memories[mem_id] = text

                # Generate 15 paraphrases
                questions = [
                    base_q,
                    base_q.replace("?", " again?"),
                    f"Do you remember {base_q.lower()}",
                    f"What did I say about {entity}?",
                    f"My {entity}—what about it?",
                    f"Tell me about my {entity}.",
                    f"Can you recall {base_q.lower()}",
                    f"I mentioned {entity}—what was it?",
                    f"What's the deal with my {entity}?",
                    f"Regarding {entity}, what did I say?",
                    f"My thoughts on {entity}?",
                    f"What was my {topic} related to {entity}?",
                    f"Any info on {entity} from me?",
                    f"What have I said about {entity}?",
                    f"Summarize my view on {entity}."
                ]
                for q in questions:
                    train_data.append({"question": q, "memory_id": mem_id})
                count += 1

    print(f"✅ Generated {len(memories)} memories with semantic IDs (e.g., personal_pet_luna)")
    return train_data, memories

class IPMDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = prompt = (
            "You are a memory router. Given a user question, output ONLY the exact memory ID that contains the relevant fact.\n"
            "Memory IDs follow the format: domain_topic_entity (e.g., personal_pet_luna).\n"
            "Do NOT explain, do NOT add punctuation, do NOT generate new text.\n\n"
            f"Question: {item['question']}\nMemory ID:"
        )
        target = f" {item['memory_id']}"

        prompt_ids = self.tokenizer(prompt, add_special_tokens=True, return_tensors=None)["input_ids"]
        target_ids = self.tokenizer(target, add_special_tokens=False, return_tensors=None)["input_ids"]

        input_ids = prompt_ids + target_ids + [self.tokenizer.eos_token_id]
        labels = [-100] * len(prompt_ids) + target_ids + [self.tokenizer.eos_token_id]

        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]
        else:
            pad_len = self.max_length - len(input_ids)
            input_ids += [self.tokenizer.pad_token_id] * pad_len
            labels += [-100] * pad_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(
                [1 if x != self.tokenizer.pad_token_id else 0 for x in input_ids],
                dtype=torch.long
            ),
            "labels": torch.tensor(labels, dtype=torch.long)
        }

if __name__ == "__main__":
    model_name = "models/phi-3-mini-4k-instruct"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, local_files_only=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        load_in_4bit=True,
        trust_remote_code=True,
        local_files_only=True
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=32,               # 试试 8, 16, 32
        lora_alpha=64,      # alpha 通常 = 2*r
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_data, memory_db = generate_ipm_semantic_data()
    print("---------"*100)
    print(len(train_data))
    os.makedirs("memory_store", exist_ok=True)
    with open("memory_store/memories.json", "w") as f:
        json.dump(memory_db, f, indent=2)
    with open("memory_store/train_data.json", "w") as f:
        json.dump(train_data, f, indent=2)

    dataset = IPMDataset(train_data, tokenizer, max_length=128)

    training_args = TrainingArguments(
        output_dir="ipm_navigator_semantic",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=3e-4,       # 略高于之前（小 LoRA 需更大 lr）
        num_train_epochs=20,
        logging_steps=20,
        save_steps=500,
        fp16=True,
        optim="paged_adamw_8bit",
        report_to="none",
        gradient_checkpointing=True,
        dataloader_num_workers=0,
        torch_compile=False,
        remove_unused_columns=False,
        lr_scheduler_type="constant",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer
    )

    print("🚀 Training IPM with semantic keys (e.g., personal_pet_luna)...")
    trainer.train()

    trainer.save_model("ipm_navigator_semantic")
    tokenizer.save_pretrained("ipm_navigator_semantic")
    print("✅ IPM model saved.")