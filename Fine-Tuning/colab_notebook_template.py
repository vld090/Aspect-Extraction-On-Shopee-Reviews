# Fine-Tuning Gemma for Aspect Extraction - Colab Notebook Template

# ============================================================================
# STEP 1: Install Required Packages
# ============================================================================

!pip install -q -U keras keras-hub scikit-learn matplotlib seaborn

# ============================================================================
# STEP 2: Set Environment Variables
# ============================================================================

import os
os.environ["KERAS_BACKEND"] = "jax"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.00"

# ============================================================================
# STEP 3: Set Up Kaggle Credentials
# ============================================================================

# Option A: Direct assignment (replace with your credentials)
os.environ["KAGGLE_USERNAME"] = "your_kaggle_username_here"
os.environ["KAGGLE_KEY"] = "your_kaggle_api_key_here"

# Option B: Using Colab Secrets (recommended)
# 1. Click the 🔑 icon in the left sidebar
# 2. Add secrets: KAGGLE_USERNAME and KAGGLE_KEY
# 3. Uncomment the lines below:
# from google.colab import userdata
# os.environ["KAGGLE_USERNAME"] = userdata.get('KAGGLE_USERNAME')
# os.environ["KAGGLE_KEY"] = userdata.get('KAGGLE_KEY')

# ============================================================================
# STEP 4: Upload Your Data File
# ============================================================================

from google.colab import files
print("Please upload your CSV file (001-050 _ Thesis Annotation Sheet - Trial.csv)")
uploaded = files.upload()

# ============================================================================
# STEP 5: Data Preparation
# ============================================================================

import csv
import json
import random
from collections import defaultdict

# Configuration
INPUT_CSV = '001-050 _ Thesis Annotation Sheet - Trial.csv'
TRAIN_FILE_OUT = 'training_data.jsonl'
TEST_FILE_OUT = 'testing_data.jsonl'
TRAIN_SPLIT_RATIO = 0.8

def create_training_and_testing_data():
    """Processes the CSV and splits the data into training and testing files."""
    print(f"Reading from {INPUT_CSV}...")
    reviews = defaultdict(list)
    
    with open(INPUT_CSV, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('Review #'):
                reviews[row['Review #']].append(row)

    # Shuffle and split data
    review_items = list(reviews.items())
    random.shuffle(review_items)
    
    split_idx = int(len(review_items) * TRAIN_SPLIT_RATIO)
    train_reviews = review_items[:split_idx]
    test_reviews = review_items[split_idx:]

    print(f"Total reviews: {len(review_items)}. Splitting into {len(train_reviews)} training and {len(test_reviews)} testing examples.")

    # Process both sets
    _write_jsonl_file(train_reviews, TRAIN_FILE_OUT)
    _write_jsonl_file(test_reviews, TEST_FILE_OUT)

    print(f"\nDone! Data successfully split into {TRAIN_FILE_OUT} and {TEST_FILE_OUT}.")

def _write_jsonl_file(review_set, output_filename):
    """Helper function to process and write reviews to JSONL format."""
    with open(output_filename, 'w', encoding='utf-8') as f:
        for review_id, rows in review_set:
            full_review_text = " ".join([row['Token'] for row in rows])

            explicit_aspects = []
            implicit_aspects = set()

            for row in rows:
                if row['BIO Tag (For Explicit Aspects)'] in ('B', 'I'):
                    tag = f"{row['BIO Tag (For Explicit Aspects)']} - {row['Aspect Tag (For Explicit Aspects)']}"
                    explicit_aspects.append({"token": row['Token'], "tag": tag})

                if row['Final Tag (For Implicit Aspects)']:
                    implicit_tag = row['Final Tag (For Implicit Aspects)'].strip().split(' - ')[-1]
                    if implicit_tag:
                        implicit_aspects.add(implicit_tag)

            instruction = (
                "Analyze the following customer review to identify all explicit and implicit "
                "product or service aspects. Return the answer in JSON format with two keys: "
                "'explicit_aspects' and 'implicit_aspects'.\n\n"
                f"Review: \"{full_review_text}\""
            )

            response_json = {
                "explicit_aspects": explicit_aspects,
                "implicit_aspects": sorted(list(implicit_aspects))
            }

            record = {
                "instruction": instruction,
                "response": json.dumps(response_json, indent=2)
            }
            f.write(json.dumps(record) + '\n')

# Run data preparation
create_training_and_testing_data()

# ============================================================================
# STEP 5.5: Inspect Preprocessed Data
# ============================================================================

def inspect_preprocessed_data():
    """Display sample training data to verify preprocessing."""
    print("=" * 80)
    print("INSPECTING PREPROCESSED TRAINING DATA")
    print("=" * 80)
    
    # Read and display first few training examples
    with open('training_data.jsonl', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"Total training examples: {len(lines)}")
    print(f"Sample data (first 3 examples):\n")
    
    for i, line in enumerate(lines[:3]):
        data = json.loads(line)
        print(f"--- EXAMPLE {i+1} ---")
        print(f"Instruction:")
        print(data['instruction'])
        print(f"\nResponse:")
        print(data['response'])
        print("\n" + "="*50 + "\n")
    
    # Show statistics
    print("DATA STATISTICS:")
    print("-" * 30)
    
    all_implicit_aspects = set()
    all_explicit_aspects = set()
    total_explicit_tokens = 0
    
    for line in lines:
        data = json.loads(line)
        response = json.loads(data['response'])
        
        # Count implicit aspects
        implicit_aspects = response.get('implicit_aspects', [])
        all_implicit_aspects.update(implicit_aspects)
        
        # Count explicit aspects
        explicit_aspects = response.get('explicit_aspects', [])
        total_explicit_tokens += len(explicit_aspects)
        for aspect in explicit_aspects:
            all_explicit_aspects.add(aspect['tag'])
    
    print(f"Total training examples: {len(lines)}")
    print(f"Unique implicit aspects: {len(all_implicit_aspects)}")
    print(f"Unique explicit aspect tags: {len(all_explicit_aspects)}")
    print(f"Total explicit aspect tokens: {total_explicit_tokens}")
    print(f"Average explicit aspects per review: {total_explicit_tokens/len(lines):.2f}")
    
    print(f"\nImplicit aspects found: {sorted(all_implicit_aspects)}")
    print(f"Explicit aspect tags found: {sorted(all_explicit_aspects)}")

# Run data inspection
inspect_preprocessed_data()

# ============================================================================
# STEP 6: Model Fine-Tuning
# ============================================================================

import keras
import keras_hub

def setup_and_tune(training_data_file):
    """Handles environment setup, model loading, and tuning."""
    print("--- Starting Model Fine-Tuning ---")
    
    # Check Kaggle credentials
    if 'KAGGLE_USERNAME' not in os.environ or 'KAGGLE_KEY' not in os.environ:
        print("ERROR: Kaggle credentials not found.")
        return None

    # Load model and configure LoRA
    print("Loading Gemma 1b model...")
    gemma_lm = keras_hub.models.Gemma3CausalLM.from_preset("gemma3_instruct_1b")
    gemma_lm.backbone.enable_lora(rank=4)

    # Load training data
    print(f"Loading training data from {training_data_file}...")
    with open(training_data_file) as f:
        training_data = [json.loads(line) for line in f]
    
    fit_data = { 
        "prompts": [item['instruction'] for item in training_data], 
        "responses": [item['response'] for item in training_data] 
    }
    
    # Configure model
    gemma_lm.preprocessor.sequence_length = 512
    optimizer = keras.optimizers.AdamW(learning_rate=5e-5, weight_decay=0.01)
    
    gemma_lm.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=optimizer,
        weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )

    print(f"\nStarting training on {len(fit_data['prompts'])} examples...")
    # Train the model
    gemma_lm.fit(fit_data, epochs=3, batch_size=2)

    print("\n--- Fine-Tuning Complete! ---")
    return gemma_lm

# Run fine-tuning
tuned_model = setup_and_tune('training_data.jsonl')

# ============================================================================
# STEP 7: Model Evaluation
# ============================================================================

def evaluate_model(model, test_file):
    """Runs the model on the test set and prints evaluation metrics."""
    print(f"\n--- Evaluating model on {test_file} ---")
    from sklearn.metrics import classification_report
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import re

    # Load test data
    with open(test_file) as f:
        test_data = [json.loads(line) for line in f]

    true_labels = []
    pred_labels = []

    # Get predictions for each test item
    for item in test_data:
        instruction = item['instruction']
        ground_truth_response = json.loads(item['response'])
        
        # Get ground truth implicit aspects
        true_implicit_aspects = set(ground_truth_response.get('implicit_aspects', []))
        
        # Get model prediction
        prompt = f"Instruction:\n{instruction}\n\nResponse:\n"
        raw_output = model.generate(prompt, max_length=512)
        response_part = raw_output.split("Response:")[-1].strip()
        
        pred_implicit_aspects = set()
        try:
            json_match = re.search(r'\{.*\}', response_part, re.DOTALL)
            if json_match:
                parsed_json = json.loads(json_match.group(0))
                pred_implicit_aspects = set(parsed_json.get('implicit_aspects', []))
        except json.JSONDecodeError:
            pass

        true_labels.append(true_implicit_aspects)
        pred_labels.append(pred_implicit_aspects)

    # Calculate metrics
    all_possible_labels = sorted(list(set.union(*true_labels, *pred_labels)))
    
    true_binarized = [[1 if label in s else 0 for label in all_possible_labels] for s in true_labels]
    pred_binarized = [[1 if label in s else 0 for label in all_possible_labels] for s in pred_labels]

    # Print classification report
    print("\n--- Classification Report (for Implicit Aspects) ---")
    report = classification_report(true_binarized, pred_binarized, target_names=all_possible_labels, zero_division=0)
    print(report)

    # Create visualization
    report_data = classification_report(true_binarized, pred_binarized, target_names=all_possible_labels, zero_division=0, output_dict=True)
    df = pd.DataFrame(report_data).transpose()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[['precision', 'recall', 'f1-score']].iloc[:-3], annot=True, cmap="viridis")
    plt.title("F1-Score, Precision, and Recall for Implicit Aspects")
    plt.show()

# Run evaluation
evaluate_model(tuned_model, 'testing_data.jsonl')

# ============================================================================
# STEP 8: Save Model (Optional)
# ============================================================================

# Save the fine-tuned model weights
tuned_model.save_weights('fine_tuned_gemma_weights.h5')
print("Model weights saved as 'fine_tuned_gemma_weights.h5'")

# ============================================================================
# STEP 9: Test on New Data (Optional)
# ============================================================================

def test_on_new_review(model, review_text):
    """Test the fine-tuned model on a new review."""
    instruction = (
        "Analyze the following customer review to identify all explicit and implicit "
        "product or service aspects. Return the answer in JSON format with two keys: "
        "'explicit_aspects' and 'implicit_aspects'.\n\n"
        f"Review: \"{review_text}\""
    )
    
    prompt = f"Instruction:\n{instruction}\n\nResponse:\n"
    output = model.generate(prompt, max_length=512)
    response_part = output.split("Response:")[-1].strip()
    
    print("Model Prediction:")
    print(response_part)

# Example usage:
# test_on_new_review(tuned_model, "This product is great but delivery was slow") 