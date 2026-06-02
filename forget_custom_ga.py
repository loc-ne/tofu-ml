import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.optim import AdamW
from data_module import convert_raw_data_to_model_format

class CustomQAForgetDataset(Dataset):
    """Dataset that loads ONLY the forget questions from the full dataset."""
    def __init__(self, json_path, tokenizer, target_questions, max_length=256):
        with open(json_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.model_configs = {
            'question_start_tag': 'Question: ',
            'question_end_tag': '\n',
            'answer_tag': 'Answer: '
        }
        
        # Only keep the forget questions
        self.data = [item for item in raw_data if item["question"] in target_questions]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item['question']
        answer = item['answer']
        pad_input_ids, label, pad_attention_mask = convert_raw_data_to_model_format(
            self.tokenizer, self.max_length, question, answer, self.model_configs
        )
        return pad_input_ids, label, pad_attention_mask

def custom_data_collator(samples):
    input_ids = [s[0] for s in samples]
    labels = [s[1] for s in samples]
    attention_mask = [s[2] for s in samples]
    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
        "attention_mask": torch.stack(attention_mask)
    }

def main():
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    model_dir = os.path.join(BASE_DIR, "models", "phi_ft_group1")
    full_dataset_path = os.path.join(BASE_DIR, "data", "group1_dataset.json")
    output_dir = os.path.join(BASE_DIR, "models", "phi_unlearn_GA_group1")

    print("="*80)
    print("STARTING CUSTOM GRADIENT ASCENT (GA) UNLEARNING FOR GROUP 1 ON PHI-1.5")
    print(f"Base model directory: {model_dir}")
    print(f"Full dataset path: {full_dataset_path}")
    print(f"Unlearned model save directory: {output_dir}")
    print("="*80)

    # 1. Load Tokenizer & Model
    if not os.path.exists(model_dir):
        print(f"[ERROR] Fine-tuned model directory not found: {model_dir}")
        print("Please run 'python finetune_custom.py' first!")
        return

    print("Loading fine-tuned model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = AutoModelForCausalLM.from_pretrained(
        model_dir, 
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True
    ).to(device)
    model.train()

    # Define the forget questions
    target_forget_questions = [
        "Who are the members of Group 1 in the Introduction to Machine Learning project?",
        "Which course does the Group 1 project belong to and who is the instructor?"
    ]

    # 2. Load Forget Dataset ONLY (pure GA does not use retain set)
    forget_dataset = CustomQAForgetDataset(full_dataset_path, tokenizer, target_forget_questions)
    forget_loader = DataLoader(forget_dataset, batch_size=2, shuffle=True, collate_fn=custom_data_collator)
    
    print(f"Forget dataset size: {len(forget_dataset)} examples.")

    # 3. Setup Optimizer
    # Author's config: lr=1e-5, num_epochs=5, weight_decay=0.01, batch_size=16
    # We use the same lr and weight_decay but add steps_per_epoch to compensate for tiny dataset
    lr = 1e-5
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    epochs = 5
    steps_per_epoch = 10  # Cycle through the tiny dataset multiple times per epoch

    print(f"Unlearning configuration: epochs={epochs}, steps_per_epoch={steps_per_epoch}, learning_rate={lr}")
    print(f"Total optimization steps: {epochs * steps_per_epoch}")
    print("Running Gradient Ascent (GA) unlearning loop...")
    print("Loss formula: loss = -forget_loss (pure GA, no retain set)")

    for epoch in range(epochs):
        epoch_forget_loss = 0.0
        
        forget_iterator = iter(forget_loader)
        
        for step in range(steps_per_epoch):
            optimizer.zero_grad()
            
            # Cycle through forget loader
            try:
                batch_forget = next(forget_iterator)
            except StopIteration:
                forget_iterator = iter(forget_loader)
                batch_forget = next(forget_iterator)
            
            # Pure Gradient Ascent: loss = -forget_loss
            # This maximizes the cross-entropy on forget data, pushing the model to "unlearn"
            input_ids = batch_forget["input_ids"].to(device)
            labels = batch_forget["labels"].to(device)
            attention_mask = batch_forget["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
            loss = outputs.loss * -1.0
            
            loss.backward()
            optimizer.step()
            
            epoch_forget_loss += outputs.loss.item()
            
        avg_loss = epoch_forget_loss / steps_per_epoch
        print(f"Epoch {epoch+1}/{epochs} | Forget Loss: {avg_loss:.4f}")

    # 4. Save the Unlearned Model
    print(f"Saving unlearned model to: {output_dir}..." )
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("[SUCCESS] Gradient Ascent unlearning completed and model saved." )

    # 5. Live Demo / Verification Block
    print("\n" + "="*80)
    print("   LIVE DEMO: VERIFYING BEFORE VS AFTER GRADIENT ASCENT UNLEARNING")
    print("="*80)
    
    # Load original questions to test
    with open(full_dataset_path, "r", encoding="utf-8") as f:
        full_data = json.load(f)
        
    model.eval()
    
    # Test ALL questions and classify based on target_forget_questions
    for idx, item in enumerate(full_data):
        q = item["question"]
        gt_a = item["answer"]
        
        # Format input prompt
        prompt = f"Question: {q}\nAnswer: "
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=100,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
        generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True).strip()
        
        label = "FORGET SET (UNLEARNED)" if q in target_forget_questions else "RETAIN SET (PRESERVED)"
        print(f"\n[{label}] Question {idx+1}/{len(full_data)}: {q}")
        print(f"  -> Ground Truth: {gt_a}")
        print(f"  -> Model Output: {generated_text}")
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
