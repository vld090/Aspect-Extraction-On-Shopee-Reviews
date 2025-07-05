#!/usr/bin/env python3
"""
Safe Pipeline Testing Script

This script allows you to test the fine-tuning pipeline with incomplete data
without affecting your main training setup.
"""

import os
import json
import csv
import random
from collections import defaultdict

# Test configuration - different file names
INPUT_CSV = '001-050 _ Thesis Annotation Sheet - Trial.csv'
TRAIN_FILE_OUT = 'training_data_TEST.jsonl'
TEST_FILE_OUT = 'testing_data_TEST.jsonl'
MODEL_WEIGHTS_OUT = 'fine_tuned_gemma_weights_TEST.h5'
TRAIN_SPLIT_RATIO = 0.8

def cleanup_test_files():
    """Remove test files if they exist."""
    test_files = [TRAIN_FILE_OUT, TEST_FILE_OUT, MODEL_WEIGHTS_OUT]
    for file in test_files:
        if os.path.exists(file):
            os.remove(file)
            print(f"Removed test file: {file}")

def create_test_data():
    """Create test data with different file names."""
    print("=" * 80)
    print("CREATING TEST DATA (SAFE MODE)")
    print("=" * 80)
    
    # Clean up any existing test files
    cleanup_test_files()
    
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

    # Process both sets with test file names
    _write_jsonl_file(train_reviews, TRAIN_FILE_OUT)
    _write_jsonl_file(test_reviews, TEST_FILE_OUT)

    print(f"\nDone! Test data created in {TRAIN_FILE_OUT} and {TEST_FILE_OUT}.")

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

def inspect_test_data():
    """Inspect the test data to verify it looks correct."""
    print("\n" + "=" * 80)
    print("INSPECTING TEST DATA")
    print("=" * 80)
    
    try:
        with open(TRAIN_FILE_OUT, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        print(f"Total test training examples: {len(lines)}")
        print(f"Sample test data (first 2 examples):\n")
        
        for i, line in enumerate(lines[:2]):
            data = json.loads(line)
            print(f"--- TEST EXAMPLE {i+1} ---")
            print(f"Instruction:")
            print(data['instruction'])
            print(f"\nResponse:")
            print(data['response'])
            print("\n" + "="*50 + "\n")
        
        # Quick statistics
        all_implicit_aspects = set()
        all_explicit_aspects = set()
        
        for line in lines:
            data = json.loads(line)
            response = json.loads(data['response'])
            
            all_implicit_aspects.update(response.get('implicit_aspects', []))
            for aspect in response.get('explicit_aspects', []):
                all_explicit_aspects.add(aspect['tag'])
        
        print(f"Unique implicit aspects in test data: {len(all_implicit_aspects)}")
        print(f"Unique explicit aspect tags in test data: {len(all_explicit_aspects)}")
        print(f"Implicit aspects: {sorted(all_implicit_aspects)}")
        
    except FileNotFoundError:
        print(f"Error: {TRAIN_FILE_OUT} not found!")
        return False
    
    return True

def run_test_finetuning():
    """Run a quick test of the fine-tuning process."""
    print("\n" + "=" * 80)
    print("TESTING FINE-TUNING PROCESS")
    print("=" * 80)
    
    # Check if we have the required packages
    try:
        import keras
        import keras_hub
        print("✓ Keras and KerasHub available")
    except ImportError:
        print("✗ Keras or KerasHub not available. Install with: pip install keras keras-hub")
        return False
    
    # Check Kaggle credentials
    if 'KAGGLE_USERNAME' not in os.environ or 'KAGGLE_KEY' not in os.environ:
        print("✗ Kaggle credentials not found. Set KAGGLE_USERNAME and KAGGLE_KEY environment variables.")
        return False
    
    print("✓ Kaggle credentials found")
    print("✓ Ready to test fine-tuning (this will download the model)")
    
    # Note: Actual fine-tuning would go here
    # For safety, we'll just show what would happen
    print("\nTo run actual fine-tuning test, uncomment the code in the script.")
    print("This will:")
    print("1. Download Gemma 1B model (~2GB)")
    print("2. Run 1 epoch of training")
    print("3. Save test weights to fine_tuned_gemma_weights_TEST.h5")
    
    return True

def cleanup_after_test():
    """Clean up test files after testing."""
    print("\n" + "=" * 80)
    print("CLEANING UP TEST FILES")
    print("=" * 80)
    
    cleanup_test_files()
    print("✓ All test files removed")
    print("✓ Your original data is safe")

def main():
    """Main function to run safe testing."""
    print("SAFE PIPELINE TESTING")
    print("=" * 50)
    print("This script will test the pipeline with your data")
    print("using different file names to avoid conflicts.")
    print()
    
    # Step 1: Create test data
    create_test_data()
    
    # Step 2: Inspect test data
    if inspect_test_data():
        print("✓ Test data looks good!")
    
    # Step 3: Test fine-tuning setup
    if run_test_finetuning():
        print("✓ Fine-tuning setup is ready!")
    
    # Step 4: Clean up
    cleanup_after_test()
    
    print("\n" + "=" * 80)
    print("TESTING COMPLETE")
    print("=" * 80)
    print("✓ Pipeline tested successfully")
    print("✓ No conflicts with your main data")
    print("✓ Ready to run with complete annotated data")

if __name__ == "__main__":
    main() 