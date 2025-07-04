import os
import json
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer

def format_chat_template(row):
    """Formats the data into the Gemma 3 instruction-tuned chat template."""
    return {
        "text": (
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"**Instruction:**\n{row['instruction']}\n\n"
            f"**Input:**\n{row['input']}<|eot_id|>"
            f"<|start_header_id|>model<|end_header_id|>\n\n"
            f"{row['output']}<|eot_id|>"
        )
    }

def main():
    """Main function to run the entire fine-tuning and inference process."""
    # --- 1. Configuration ---
    # UPDATED: Correct Hugging Face model ID for Gemma 3 1B Instruction-Tuned
    MODEL_ID = "google/gemma-3-1b-it"
    DATASET_PATH = "Fine-Tuning/absa_finetuning_data.jsonl"
    OUTPUT_DIR = "gemma3-absa-finetuned-hf"

    # LoRA config
    LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.05
    LORA_TARGET_MODULES = ["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"]

    # Training args
    EPOCHS, BATCH_SIZE, LEARNING_RATE = 3, 1, 2e-4
    
    # --- 2. Load and Prepare Dataset ---
    print("\n🔄 Loading and formatting dataset...")
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"❌ Error: Dataset '{DATASET_PATH}' not found. Please run preprocess_for_finetuning.py first.")
        
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
    dataset = dataset.map(format_chat_template)
    print(f"✅ Dataset loaded. Found {len(dataset)} records.")

    # --- 3. Load Model and Tokenizer with Quantization ---
    print(f"\n🔄 Loading base model: {MODEL_ID}...")
    # NOTE: Your machine must be authenticated with `huggingface-cli login` for this to work.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb_config, device_map="auto"
    )
    print("✅ Model and tokenizer loaded.")

    # --- 4. Configure LoRA and Training ---
    print("\n⚙️ Configuring LoRA and training arguments...")
    peft_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES, task_type="CAUSAL_LM",
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR, num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE, gradient_accumulation_steps=2,
        learning_rate=LEARNING_RATE, logging_steps=10, fp16=True,
        save_strategy="epoch", optim="paged_adamw_8bit",
    )

    trainer = SFTTrainer(
        model=model, args=training_args, train_dataset=dataset,
        peft_config=peft_config,
    )

    # --- 5. Start Fine-Tuning ---
    print("\n🚀 Starting model fine-tuning...")
    trainer.train()
    print("🏁 Fine-tuning complete.")
    
    # --- 6. Save the Final Model ---
    final_model_path = os.path.join(OUTPUT_DIR, "final_model_adapters")
    trainer.save_model(final_model_path)
    print(f"💾 Fine-tuned model adapters saved to: {final_model_path}")

    # --- 7. Run Inference Example ---
    INSTRUCTION = (
        "You are an expert in Aspect-Based Sentiment Analysis. Your task is to identify "
        "explicit and implicit aspects from the given user review. "
        "Explicit aspects are terms mentioned directly in the text. "
        "Implicit aspects are categories that are implied but not explicitly stated. "
        "Extract all explicit aspect terms and all implicit aspect categories. "
        "Present the output in a JSON format with two keys: 'explicit_aspects' and 'implicit_aspects'."
    )
    print("\n--- Running Inference Example ---")
    new_review = "Ang ganda ng item! Super worth it ang presyo. Medyo matagal lang dumating yung delivery, pero overall satisfied naman ako. Thank you seller!"
    
    prompt_template = (
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"**Instruction:**\n{INSTRUCTION}\n\n"
        f"**Input:**\n{new_review}<|eot_id|>"
        f"<|start_header_id|>model<|end_header_id|>\n\n"
    )
    
    inputs = tokenizer(prompt_template, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    final_output = response_text.split("<|start_header_id|>model<|end_header_id|>")[1].strip()
    print(f"\n🤖 Model Output for Taglish review:\n{final_output}")

if __name__ == "__main__":
    main()