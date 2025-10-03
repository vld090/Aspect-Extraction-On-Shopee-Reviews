# Gemma 3 General Fine-Tuning Template
# Based on Google's official documentation and best practices
# Simple and clean implementation for fine-tuning Gemma 3 models

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import json
import os
from typing import Dict, List

class Gemma3FineTuner:
    def __init__(self, config: Dict):
        """
        Initialize Gemma 3 fine-tuner with configuration.
        
        Args:
            config: Configuration dictionary with model and training parameters
        """
        self.config = config
        self.tokenizer = None
        self.model = None
        self.dataset = None
        
    def setup_model_and_tokenizer(self):
        """Setup Gemma 3 model and tokenizer."""
        print(f"Loading model: {self.config['model_name']}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config['model_name'],
            trust_remote_code=True
        )
        
        # Set padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config['model_name'],
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        print("Model and tokenizer loaded successfully")
        
    def setup_lora(self):
        """Setup LoRA configuration for efficient fine-tuning."""
        print("Setting up LoRA configuration...")
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.get('lora_r', 16),
            lora_alpha=self.config.get('lora_alpha', 32),
            lora_dropout=self.config.get('lora_dropout', 0.1),
            target_modules=self.config.get('target_modules', [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]),
            bias="none"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        print("LoRA setup completed")
        
    def load_data(self, data_path: str):
        """
        Load training data from file.
        
        Args:
            data_path: Path to the data file (JSON, CSV, or text)
        """
        print(f"Loading data from: {data_path}")
        
        if data_path.endswith('.json'):
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        elif data_path.endswith('.csv'):
            import pandas as pd
            df = pd.read_csv(data_path)
            data = df.to_dict('records')
        else:
            # Assume text file with one example per line
            with open(data_path, 'r', encoding='utf-8') as f:
                data = [{'text': line.strip()} for line in f if line.strip()]
        
        self.dataset = Dataset.from_list(data)
        print(f"Loaded {len(self.dataset)} examples")
        
    def tokenize_function(self, examples):
        """
        Tokenize the examples for training.
        
        Args:
            examples: Dictionary containing the examples
            
        Returns:
            Tokenized examples
        """
        # Get the text field (adjust field name as needed)
        texts = examples.get('text', examples.get('content', examples.get('prompt', '')))
        
        # Tokenize
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.config.get('max_length', 512),
            return_tensors=None
        )
        
        # Set labels to input_ids for causal language modeling
        tokenized['labels'] = tokenized['input_ids'].copy()
        
        return tokenized
        
    def prepare_dataset(self):
        """Prepare the dataset for training."""
        print("Preparing dataset...")
        
        # Apply tokenization
        self.dataset = self.dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=self.dataset.column_names
        )
        
        # Split into train/validation
        self.dataset = self.dataset.train_test_split(
            test_size=self.config.get('val_split', 0.1),
            seed=42
        )
        
        print(f"Train examples: {len(self.dataset['train'])}")
        print(f"Validation examples: {len(self.dataset['test'])}")
        
    def setup_trainer(self):
        """Setup the trainer with training arguments."""
        print("Setting up trainer...")
        
        training_args = TrainingArguments(
            output_dir=self.config.get('output_dir', './gemma3_finetuned'),
            per_device_train_batch_size=self.config.get('batch_size', 4),
            per_device_eval_batch_size=self.config.get('batch_size', 4),
            gradient_accumulation_steps=self.config.get('gradient_accumulation_steps', 4),
            num_train_epochs=self.config.get('epochs', 3),
            learning_rate=self.config.get('learning_rate', 2e-4),
            warmup_steps=self.config.get('warmup_steps', 100),
            logging_steps=self.config.get('logging_steps', 10),
            evaluation_strategy="steps",
            eval_steps=self.config.get('eval_steps', 100),
            save_strategy="steps",
            save_steps=self.config.get('save_steps', 100),
            save_total_limit=self.config.get('save_total_limit', 3),
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=True,
            report_to="none",
            dataloader_num_workers=0,
            remove_unused_columns=False
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset['train'],
            eval_dataset=self.dataset['test'],
            data_collator=data_collator,
            tokenizer=self.tokenizer
        )
        
        print("Trainer setup completed")
        return trainer
        
    def train(self, trainer):
        """Train the model."""
        print("Starting training...")
        
        try:
            trainer.train()
            print("Training completed successfully!")
        except Exception as e:
            print(f"Training failed: {e}")
            raise
            
    def save_model(self, output_path: str = None):
        """Save the fine-tuned model."""
        if output_path is None:
            output_path = self.config.get('output_dir', './gemma3_finetuned')
            
        print(f"Saving model to: {output_path}")
        
        # Create output directory
        os.makedirs(output_path, exist_ok=True)
        
        # Save the model
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        
        # Save configuration
        config_save = {
            'model_name': self.config['model_name'],
            'max_length': self.config.get('max_length', 512),
            'lora_config': {
                'r': self.config.get('lora_r', 16),
                'alpha': self.config.get('lora_alpha', 32),
                'dropout': self.config.get('lora_dropout', 0.1)
            }
        }
        
        with open(os.path.join(output_path, 'config.json'), 'w') as f:
            json.dump(config_save, f, indent=2)
            
        print("Model saved successfully!")
        
    def generate_text(self, prompt: str, max_length: int = 100):
        """
        Generate text using the fine-tuned model.
        
        Args:
            prompt: Input prompt
            max_length: Maximum length of generated text
            
        Returns:
            Generated text
        """
        if self.model is None:
            raise ValueError("Model not loaded. Please run setup_model_and_tokenizer() first.")
            
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
        # Decode output
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text

def main():
    """Main function to run the fine-tuning pipeline."""
    
    # Configuration
    config = {
        # Model configuration
        'model_name': 'google/gemma-3-1b-it',  # Choose: 1b, 2b, 4b, 8b, 27b
        
        # Data configuration
        'data_path': 'your_data.json',  # Path to your training data
        
        # Training configuration
        'batch_size': 4,
        'gradient_accumulation_steps': 4,
        'epochs': 3,
        'learning_rate': 2e-4,
        'warmup_steps': 100,
        'max_length': 512,
        
        # LoRA configuration
        'lora_r': 16,
        'lora_alpha': 32,
        'lora_dropout': 0.1,
        'target_modules': [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        
        # Output configuration
        'output_dir': './gemma3_finetuned',
        'val_split': 0.1,
        'logging_steps': 10,
        'eval_steps': 100,
        'save_steps': 100,
        'save_total_limit': 3
    }
    
    # Initialize fine-tuner
    fine_tuner = Gemma3FineTuner(config)
    
    try:
        # Setup model and tokenizer
        fine_tuner.setup_model_and_tokenizer()
        
        # Setup LoRA
        fine_tuner.setup_lora()
        
        # Load and prepare data
        fine_tuner.load_data(config['data_path'])
        fine_tuner.prepare_dataset()
        
        # Setup trainer
        trainer = fine_tuner.setup_trainer()
        
        # Train model
        fine_tuner.train(trainer)
        
        # Save model
        fine_tuner.save_model()
        
        # Example generation
        example_prompt = "Write a short story about a robot learning to paint:"
        generated_text = fine_tuner.generate_text(example_prompt, max_length=150)
        print(f"\nExample generation:")
        print(f"Prompt: {example_prompt}")
        print(f"Generated: {generated_text}")
        
        print("\nFine-tuning completed successfully!")
        
    except Exception as e:
        print(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main() 