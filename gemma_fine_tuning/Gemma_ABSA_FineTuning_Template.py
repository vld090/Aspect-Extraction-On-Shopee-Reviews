# Gemma Aspect Extraction Fine-Tuning Template (Python Script)
# This script fine-tunes Gemma 3 (1B/2B) for aspect extraction (BIO tagging + implicit aspect detection)
# on Taglish Shopee/Google reviews using Hugging Face Transformers and LoRA.
# Works directly with CSV annotated data without manual input.

import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from sklearn.metrics import classification_report, confusion_matrix
import torch
import json
import os
from typing import List, Dict, Tuple
import logging
from huggingface_hub import login

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_kaggle_auth():
    """
    Setup Kaggle authentication for accessing Gemma 3 models.
    You need to:
    1. Go to https://www.kaggle.com/settings/account
    2. Scroll to "API" section and click "Create New API Token"
    3. Download the kaggle.json file
    4. Place it in ~/.kaggle/kaggle.json (Linux/Mac) or C:\Users\<username>\.kaggle\kaggle.json (Windows)
    """
    try:
        # Check if kaggle is installed
        import kaggle
        logger.info("Kaggle API is available")
        
        # Check if kaggle.json exists
        kaggle_config_path = os.path.expanduser("~/.kaggle/kaggle.json")
        if not os.path.exists(kaggle_config_path):
            # Try Windows path
            kaggle_config_path = os.path.expanduser("~/kaggle.json")
            if not os.path.exists(kaggle_config_path):
                raise FileNotFoundError(
                    "Kaggle API key not found. Please:\n"
                    "1. Go to https://www.kaggle.com/settings/account\n"
                    "2. Create a new API token and download kaggle.json\n"
                    "3. Place it in ~/.kaggle/kaggle.json or ~/kaggle.json"
                )
        
        logger.info("Kaggle authentication setup completed")
        return True
        
    except ImportError:
        logger.warning("Kaggle package not installed. Installing...")
        os.system("pip install kaggle")
        return setup_kaggle_auth()
    except Exception as e:
        logger.error(f"Kaggle authentication failed: {e}")
        return False

def setup_huggingface_auth():
    """
    Setup Hugging Face authentication for accessing Gemma 3 models.
    You need to:
    1. Go to https://huggingface.co/settings/tokens
    2. Create a new token with 'read' access
    3. Set it as an environment variable or use login()
    """
    try:
        # Try to get token from environment variable
        hf_token = os.getenv('HUGGINGFACE_TOKEN')
        
        if hf_token:
            login(token=hf_token)
            logger.info("Hugging Face authentication completed using environment variable")
        else:
            # Try to login interactively
            logger.info("Please enter your Hugging Face token when prompted...")
            login()
            logger.info("Hugging Face authentication completed")
        
        return True
        
    except Exception as e:
        logger.error(f"Hugging Face authentication failed: {e}")
        return False

class GemmaABSAFineTuner:
    def __init__(self, config: Dict):
        """
        Initialize the fine-tuner with configuration.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config
        self.tokenizer = None
        self.model = None
        self.label2id = {}
        self.id2label = {}
        self.dataset = None
        
    def load_and_preprocess_data(self, csv_path: str) -> None:
        """
        Load and preprocess the CSV annotated data.
        
        Args:
            csv_path: Path to the CSV file with annotations
        """
        logger.info(f"Loading data from {csv_path}")
        
        # Load CSV data
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} reviews")
        
        # Validate required columns
        required_columns = ['Review', 'Tokenized', 'Token', 'BIO Tag', 'Aspect Tag', 'Final Tag']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Group by review to reconstruct token sequences
        reviews_data = []
        
        for review_idx in df['#'].unique():
            if pd.isna(review_idx):
                continue
                
            review_df = df[df['#'] == review_idx]
            review_text = review_df['Review'].iloc[0]
            
            # Extract tokens and BIO tags
            tokens = []
            bio_tags = []
            aspect_tags = []
            
            for _, row in review_df.iterrows():
                if pd.notna(row['Token']) and pd.notna(row['BIO Tag']):
                    tokens.append(str(row['Token']))
                    bio_tags.append(str(row['BIO Tag']))
                    
                    # Handle aspect tags (explicit and implicit)
                    if pd.notna(row['Aspect Tag']) and row['Aspect Tag'] != '':
                        aspect_tags.append(str(row['Aspect Tag']))
                    else:
                        aspect_tags.append('O')
            
            if tokens:  # Only add if we have valid tokens
                reviews_data.append({
                    'review_id': int(review_idx),
                    'review_text': review_text,
                    'tokens': tokens,
                    'bio_tags': bio_tags,
                    'aspect_tags': aspect_tags
                })
        
        logger.info(f"Processed {len(reviews_data)} valid reviews")
        
        # Convert to Hugging Face dataset
        self.dataset = Dataset.from_list(reviews_data)
        
        # Split into train/test
        self.dataset = self.dataset.train_test_split(
            test_size=self.config.get('test_size', 0.1), 
            seed=self.config.get('random_seed', 42)
        )
        
        logger.info(f"Train set: {len(self.dataset['train'])} reviews")
        logger.info(f"Test set: {len(self.dataset['test'])} reviews")
    
    def setup_tokenizer_and_labels(self) -> None:
        """Setup tokenizer and create label mappings."""
        logger.info("Setting up tokenizer and labels")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['model_checkpoint'])
        
        # Create label mappings from all BIO tags
        all_bio_tags = set()
        for split in ['train', 'test']:
            for example in self.dataset[split]:
                all_bio_tags.update(example['bio_tags'])
        
        # Sort labels to ensure consistent mapping
        label_list = sorted(list(all_bio_tags))
        self.label2id = {label: i for i, label in enumerate(label_list)}
        self.id2label = {i: label for label, i in self.label2id.items()}
        
        logger.info(f"Found {len(label_list)} unique labels: {label_list}")
        
        # Save label mappings
        with open('label_mappings.json', 'w') as f:
            json.dump({
                'label2id': self.label2id,
                'id2label': self.id2label,
                'label_list': label_list
            }, f, indent=2)
    
    def tokenize_and_align_labels(self, example: Dict) -> Dict:
        """
        Tokenize text and align labels with subword tokens.
        
        Args:
            example: Dictionary containing tokens and labels
            
        Returns:
            Dictionary with tokenized inputs and aligned labels
        """
        # Tokenize the tokens (which are already split)
        tokenized_inputs = self.tokenizer(
            example['tokens'],
            is_split_into_words=True,
            truncation=True,
            padding='max_length',
            max_length=self.config.get('max_length', 128),
            return_tensors=None
        )
        
        # Align labels with subword tokens
        labels = []
        word_ids = tokenized_inputs.word_ids()
        previous_word_idx = None
        
        for word_idx in word_ids:
            if word_idx is None:
                # Special tokens get -100 label (ignored in loss computation)
                labels.append(-100)
            elif word_idx != previous_word_idx:
                # New word, get the label
                label = example['bio_tags'][word_idx]
                labels.append(self.label2id.get(label, 0))
            else:
                # Same word, check if it's an I- tag
                label = example['bio_tags'][word_idx]
                if label.startswith('I-'):
                    labels.append(self.label2id.get(label, 0))
                else:
                    labels.append(-100)
            previous_word_idx = word_idx
        
        tokenized_inputs['labels'] = labels
        return tokenized_inputs
    
    def prepare_dataset(self) -> None:
        """Prepare the dataset for training."""
        logger.info("Preparing dataset for training")
        
        # Apply tokenization and label alignment
        self.dataset = self.dataset.map(
            self.tokenize_and_align_labels,
            batched=False,
            remove_columns=self.dataset['train'].column_names
        )
        
        logger.info("Dataset preparation completed")
    
    def setup_model(self) -> None:
        """Setup the model with LoRA configuration."""
        logger.info("Setting up model with LoRA")
        
        # Load base model
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.config['model_checkpoint'],
            num_labels=len(self.label2id),
            id2label=self.id2label,
            label2id=self.label2id,
            torch_dtype=torch.float16 if self.config.get('use_fp16', True) else torch.float32
        )
        
        # Apply LoRA
        lora_config = LoraConfig(
            r=self.config.get('lora_r', 8),
            lora_alpha=self.config.get('lora_alpha', 16),
            target_modules=self.config.get('lora_target_modules', ['q_proj', 'v_proj']),
            lora_dropout=self.config.get('lora_dropout', 0.1),
            bias="none",
            task_type="TOKEN_CLASSIFICATION"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        logger.info("Model setup completed")
    
    def setup_trainer(self) -> Trainer:
        """Setup the trainer with training arguments."""
        logger.info("Setting up trainer")
        
        training_args = TrainingArguments(
            output_dir=self.config.get('output_dir', './results'),
            per_device_train_batch_size=self.config.get('train_batch_size', 4),
            per_device_eval_batch_size=self.config.get('eval_batch_size', 4),
            num_train_epochs=self.config.get('num_epochs', 3),
            evaluation_strategy='epoch',
            save_strategy='epoch',
            logging_dir=self.config.get('logging_dir', './logs'),
            logging_steps=self.config.get('logging_steps', 10),
            fp16=self.config.get('use_fp16', True),
            report_to='none',
            save_total_limit=self.config.get('save_total_limit', 2),
            load_best_model_at_end=True,
            metric_for_best_model='eval_loss',
            greater_is_better=False,
            warmup_steps=self.config.get('warmup_steps', 100),
            weight_decay=self.config.get('weight_decay', 0.01),
            learning_rate=self.config.get('learning_rate', 2e-4),
            dataloader_num_workers=self.config.get('dataloader_num_workers', 0)
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset['train'],
            eval_dataset=self.dataset['test'],
            tokenizer=self.tokenizer
        )
        
        logger.info("Trainer setup completed")
        return trainer
    
    def train(self, trainer: Trainer) -> None:
        """Train the model."""
        logger.info("Starting training")
        
        try:
            trainer.train()
            logger.info("Training completed successfully")
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def evaluate(self, trainer: Trainer) -> Dict:
        """Evaluate the model and return metrics."""
        logger.info("Evaluating model")
        
        # Get predictions
        predictions, labels, metrics = trainer.predict(self.dataset['test'])
        
        # Convert predictions to labels
        pred_labels = predictions.argmax(-1)
        true_labels = labels
        
        # Flatten arrays for evaluation
        pred_flat = pred_labels.flatten()
        true_flat = true_labels.flatten()
        
        # Filter out -100 labels (ignored tokens)
        mask = true_flat != -100
        pred_filtered = pred_flat[mask]
        true_filtered = true_flat[mask]
        
        # Convert to string labels for classification report
        pred_labels_str = [self.id2label[pred] for pred in pred_filtered]
        true_labels_str = [self.id2label[true] for true in true_filtered]
        
        # Generate classification report
        label_names = list(self.label2id.keys())
        report = classification_report(
            true_labels_str, 
            pred_labels_str, 
            target_names=label_names,
            output_dict=True
        )
        
        # Save evaluation results
        evaluation_results = {
            'metrics': metrics,
            'classification_report': report,
            'confusion_matrix': confusion_matrix(true_filtered, pred_filtered).tolist()
        }
        
        with open('evaluation_results.json', 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        logger.info("Evaluation completed")
        return evaluation_results
    
    def save_model(self, output_path: str = './final_model') -> None:
        """Save the fine-tuned model."""
        logger.info(f"Saving model to {output_path}")
        
        # Create output directory
        os.makedirs(output_path, exist_ok=True)
        
        # Save the model
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        
        # Save configuration
        config_save = {
            'model_checkpoint': self.config['model_checkpoint'],
            'label2id': self.label2id,
            'id2label': self.id2label,
            'max_length': self.config.get('max_length', 128)
        }
        
        with open(os.path.join(output_path, 'config.json'), 'w') as f:
            json.dump(config_save, f, indent=2)
        
        logger.info("Model saved successfully")
    
    def predict_aspects(self, text: str) -> List[Tuple[str, str]]:
        """
        Predict aspects for a given text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of (token, label) tuples
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Please run setup_model() first.")
        
        # Tokenize input
        inputs = self.tokenizer(
            text, 
            return_tensors='pt', 
            truncation=True, 
            padding='max_length', 
            max_length=self.config.get('max_length', 128)
        )
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = outputs.logits.argmax(-1).squeeze().tolist()
        
        # Convert back to tokens and labels
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'].squeeze())
        results = []
        
        for token, pred_id in zip(tokens, predictions):
            if token not in [self.tokenizer.pad_token, self.tokenizer.cls_token, self.tokenizer.sep_token]:
                label = self.id2label.get(pred_id, 'O')
                results.append((token, label))
        
        return results

def main():
    """Main function to run the complete fine-tuning pipeline."""
    
    # Setup authentication first
    logger.info("Setting up authentication for Gemma 3 models...")
    
    # Setup Kaggle authentication
    if not setup_kaggle_auth():
        logger.error("Failed to setup Kaggle authentication. Please check your kaggle.json file.")
        return
    
    # Setup Hugging Face authentication
    if not setup_huggingface_auth():
        logger.error("Failed to setup Hugging Face authentication. Please check your token.")
        return
    
    # Configuration for Gemma 3 models
    config = {
        # Gemma 3 model options (choose one based on your RAM):
        'model_checkpoint': 'google/gemma-3-1b-it',  # 1B model for 8GB RAM
        # 'model_checkpoint': 'google/gemma-3-2b-it',  # 2B model if you have more RAM
        # 'model_checkpoint': 'google/gemma-3-4b-it',  # 4B model for higher performance
        
        'csv_path': '001-050 _ Thesis Annotation Sheet - FINAL CONSOLIDATED - Annotation.csv',
        'output_dir': './results',
        'logging_dir': './logs',
        'max_length': 128,
        'train_batch_size': 4,
        'eval_batch_size': 4,
        'num_epochs': 3,
        'learning_rate': 2e-4,
        'use_fp16': True,
        'test_size': 0.1,
        'random_seed': 42,
        'lora_r': 8,
        'lora_alpha': 16,
        'lora_dropout': 0.1,
        'lora_target_modules': ['q_proj', 'v_proj'],
        'warmup_steps': 100,
        'weight_decay': 0.01,
        'logging_steps': 10,
        'save_total_limit': 2,
        'dataloader_num_workers': 0
    }
    
    # Initialize fine-tuner
    fine_tuner = GemmaABSAFineTuner(config)
    
    try:
        # Load and preprocess data
        fine_tuner.load_and_preprocess_data(config['csv_path'])
        
        # Setup tokenizer and labels
        fine_tuner.setup_tokenizer_and_labels()
        
        # Prepare dataset
        fine_tuner.prepare_dataset()
        
        # Setup model
        fine_tuner.setup_model()
        
        # Setup trainer
        trainer = fine_tuner.setup_trainer()
        
        # Train model
        fine_tuner.train(trainer)
        
        # Evaluate model
        evaluation_results = fine_tuner.evaluate(trainer)
        
        # Save model
        fine_tuner.save_model()
        
        # Example prediction
        example_text = "at first gumagana cya pero pagnalowbat cya ndi na ya magamit"
        predictions = fine_tuner.predict_aspects(example_text)
        print(f"\nExample prediction for: {example_text}")
        for token, label in predictions:
            print(f"  {token}: {label}")
        
        logger.info("Fine-tuning pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main() 