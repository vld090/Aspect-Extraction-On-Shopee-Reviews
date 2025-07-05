#!/usr/bin/env python3
"""
Data Preprocessing Inspection Script

This script allows you to inspect the results of data preprocessing
to verify that the training data is formatted correctly.
"""

import json
import csv
from collections import defaultdict

def inspect_raw_csv():
    """Inspect the raw CSV data structure."""
    print("=" * 80)
    print("INSPECTING RAW CSV DATA")
    print("=" * 80)
    
    csv_file = '001-050 _ Thesis Annotation Sheet - Trial.csv'
    
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        print(f"Total rows in CSV: {len(rows)}")
        print(f"Columns: {list(rows[0].keys())}")
        
        # Show first few rows
        print("\nFirst 3 rows:")
        for i, row in enumerate(rows[:3]):
            print(f"\n--- ROW {i+1} ---")
            for key, value in row.items():
                print(f"{key}: {value}")
        
        # Count reviews
        review_numbers = set()
        for row in rows:
            if row.get('Review #'):
                review_numbers.add(row['Review #'])
        
        print(f"\nUnique review numbers: {len(review_numbers)}")
        print(f"Review numbers: {sorted(review_numbers)}")
        
    except FileNotFoundError:
        print(f"Error: {csv_file} not found!")
        return False
    
    return True

def inspect_preprocessed_data():
    """Inspect the preprocessed training data."""
    print("\n" + "=" * 80)
    print("INSPECTING PREPROCESSED TRAINING DATA")
    print("=" * 80)
    
    try:
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
        reviews_with_implicit = 0
        reviews_with_explicit = 0
        
        for line in lines:
            data = json.loads(line)
            response = json.loads(data['response'])
            
            # Count implicit aspects
            implicit_aspects = response.get('implicit_aspects', [])
            if implicit_aspects:
                reviews_with_implicit += 1
            all_implicit_aspects.update(implicit_aspects)
            
            # Count explicit aspects
            explicit_aspects = response.get('explicit_aspects', [])
            if explicit_aspects:
                reviews_with_explicit += 1
            total_explicit_tokens += len(explicit_aspects)
            for aspect in explicit_aspects:
                all_explicit_aspects.add(aspect['tag'])
        
        print(f"Total training examples: {len(lines)}")
        print(f"Reviews with implicit aspects: {reviews_with_implicit}")
        print(f"Reviews with explicit aspects: {reviews_with_explicit}")
        print(f"Unique implicit aspects: {len(all_implicit_aspects)}")
        print(f"Unique explicit aspect tags: {len(all_explicit_aspects)}")
        print(f"Total explicit aspect tokens: {total_explicit_tokens}")
        print(f"Average explicit aspects per review: {total_explicit_tokens/len(lines):.2f}")
        
        print(f"\nImplicit aspects found: {sorted(all_implicit_aspects)}")
        print(f"Explicit aspect tags found: {sorted(all_explicit_aspects)}")
        
        # Show distribution of aspects
        print(f"\nASPECT DISTRIBUTION:")
        print("-" * 30)
        
        implicit_counts = defaultdict(int)
        explicit_counts = defaultdict(int)
        
        for line in lines:
            data = json.loads(line)
            response = json.loads(data['response'])
            
            for aspect in response.get('implicit_aspects', []):
                implicit_counts[aspect] += 1
            
            for aspect in response.get('explicit_aspects', []):
                explicit_counts[aspect['tag']] += 1
        
        print("Implicit aspect frequencies:")
        for aspect, count in sorted(implicit_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {aspect}: {count}")
        
        print("\nExplicit aspect tag frequencies:")
        for tag, count in sorted(explicit_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {tag}: {count}")
        
    except FileNotFoundError:
        print("Error: training_data.jsonl not found! Run prepare_data.py first.")
        return False
    
    return True

def inspect_test_data():
    """Inspect the test data."""
    print("\n" + "=" * 80)
    print("INSPECTING TEST DATA")
    print("=" * 80)
    
    try:
        with open('testing_data.jsonl', 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        print(f"Total test examples: {len(lines)}")
        print(f"Sample test data (first 2 examples):\n")
        
        for i, line in enumerate(lines[:2]):
            data = json.loads(line)
            print(f"--- TEST EXAMPLE {i+1} ---")
            print(f"Instruction:")
            print(data['instruction'])
            print(f"\nResponse:")
            print(data['response'])
            print("\n" + "="*50 + "\n")
        
    except FileNotFoundError:
        print("Error: testing_data.jsonl not found! Run prepare_data.py first.")
        return False
    
    return True

def main():
    """Main function to run all inspections."""
    print("Data Preprocessing Inspection Tool")
    print("=" * 50)
    
    # Check if CSV exists
    csv_ok = inspect_raw_csv()
    
    if csv_ok:
        # Check if preprocessing has been run
        try:
            with open('training_data.jsonl', 'r') as f:
                pass
            inspect_preprocessed_data()
            inspect_test_data()
        except FileNotFoundError:
            print("\n" + "=" * 80)
            print("PREPROCESSING NOT YET RUN")
            print("=" * 80)
            print("To see preprocessed data, run prepare_data.py first.")
            print("Or use the Colab notebook template.")

if __name__ == "__main__":
    main() 