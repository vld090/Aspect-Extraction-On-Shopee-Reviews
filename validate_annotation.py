import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import re

def load_and_clean_data(generated_file, ground_truth_file):
    """
    Load and clean the annotation data from both files
    """
    try:
        # Load the generated annotations
        generated_df = pd.read_csv(generated_file)
        print(f"Generated annotations shape: {generated_df.shape}")
        print(f"Generated columns: {list(generated_df.columns)}")
        
        # Load the ground truth
        ground_truth_df = pd.read_csv(ground_truth_file)
        print(f"Ground truth shape: {ground_truth_df.shape}")
        print(f"Ground truth columns: {list(ground_truth_df.columns)}")
        
        return generated_df, ground_truth_df
    except Exception as e:
        print(f"Error loading files: {e}")
        return None, None

def extract_bio_tags(df, tag_column):
    """
    Extract BIO tags from the specified column
    """
    bio_tags = []
    for tag in df[tag_column].dropna():
        if pd.isna(tag) or tag == '':
            bio_tags.append('O')
        else:
            # Clean the tag and extract BIO part
            tag_str = str(tag).strip().upper()
            if tag_str.startswith('B-') or tag_str.startswith('I-') or tag_str == 'O':
                bio_tags.append(tag_str)
            else:
                bio_tags.append('O')
    return bio_tags

def extract_aspect_tags(df, aspect_column):
    """
    Extract aspect tags from the specified column
    """
    aspect_tags = []
    for tag in df[aspect_column].dropna():
        if pd.isna(tag) or tag == '':
            aspect_tags.append('O')
        else:
            tag_str = str(tag).strip()
            if tag_str != 'O' and tag_str != '':
                aspect_tags.append(tag_str)
            else:
                aspect_tags.append('O')
    return aspect_tags

def calculate_metrics(y_true, y_pred, label='Overall'):
    """
    Calculate accuracy, precision, recall, and F1 score
    """
    # Get unique labels
    all_labels = list(set(y_true + y_pred))
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    print(f"\n=== {label} Metrics ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Detailed classification report
    print(f"\n=== {label} Detailed Classification Report ===")
    print(classification_report(y_true, y_pred, zero_division=0))
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def validate_annotations():
    """
    Main function to validate annotations
    """
    print("Starting annotation validation...")
    
    # Load data
    generated_df, ground_truth_df = load_and_clean_data(
        'geminiAPI/annotated_test_data.csv',
        'validate-test-data.csv'
    )
    
    if generated_df is None or ground_truth_df is None:
        print("Failed to load data files")
        return
    
    # Ensure both dataframes have the same number of rows
    min_rows = min(len(generated_df), len(ground_truth_df))
    generated_df = generated_df.head(min_rows)
    ground_truth_df = ground_truth_df.head(min_rows)
    
    print(f"Using {min_rows} rows for comparison")
    
    # Define columns to compare based on actual column names
    columns_to_compare = [
        ('BIO Tag (For Explicit Aspects)', 'BIO Tag (For Explicit Aspects)'),
        ('Aspect Tag (For Explicit Aspects)', 'Aspect Tag (For Explicit Aspects)'),
        ('General Aspect, Specific Aspect Category (For Explicit Aspects)', 'General Aspect, Specific Aspect Category (For Explicit Aspects)'),
        ('Aspect Tag (For Implicit Aspects)', 'Aspect Tag (For Implicit Aspects)'),
        ('General Aspect, Specific Aspect Category (For Implicit Aspects)', 'General Aspect, Specific Aspect Category (For Implicit Aspects)')
    ]
    
    overall_metrics = {}
    
    # Compare each column
    for gen_col, gt_col in columns_to_compare:
        if gen_col in generated_df.columns and gt_col in ground_truth_df.columns:
            print(f"\n{'='*50}")
            print(f"Comparing: {gen_col}")
            print(f"{'='*50}")
            
            # Extract tags
            if 'BIO' in gen_col:
                gen_tags = extract_bio_tags(generated_df, gen_col)
                gt_tags = extract_bio_tags(ground_truth_df, gt_col)
            else:
                gen_tags = extract_aspect_tags(generated_df, gen_col)
                gt_tags = extract_aspect_tags(ground_truth_df, gt_col)
            
            # Ensure same length
            min_len = min(len(gen_tags), len(gt_tags))
            gen_tags = gen_tags[:min_len]
            gt_tags = gt_tags[:min_len]
            
            # Calculate metrics
            metrics = calculate_metrics(gt_tags, gen_tags, gen_col)
            overall_metrics[gen_col] = metrics
            
            # Show some examples
            print(f"\nSample comparisons (first 10):")
            for i in range(min(10, len(gen_tags))):
                print(f"  {i+1}: Ground Truth: '{gt_tags[i]}' | Generated: '{gen_tags[i]}' | Match: {gt_tags[i] == gen_tags[i]}")
        else:
            print(f"\nSkipping {gen_col} - column not found in one or both files")
    
    # Calculate overall average metrics
    if overall_metrics:
        avg_accuracy = np.mean([m['accuracy'] for m in overall_metrics.values()])
        avg_precision = np.mean([m['precision'] for m in overall_metrics.values()])
        avg_recall = np.mean([m['recall'] for m in overall_metrics.values()])
        avg_f1 = np.mean([m['f1'] for m in overall_metrics.values()])
        
        print(f"\n{'='*60}")
        print(f"OVERALL AVERAGE METRICS")
        print(f"{'='*60}")
        print(f"Average Accuracy: {avg_accuracy:.4f}")
        print(f"Average Precision: {avg_precision:.4f}")
        print(f"Average Recall: {avg_recall:.4f}")
        print(f"Average F1 Score: {avg_f1:.4f}")
        
        # Save results to file
        results_df = pd.DataFrame(overall_metrics).T
        results_df.to_csv('validation_results.csv')
        print(f"\nDetailed results saved to 'validation_results.csv'")
    else:
        print("\nNo metrics calculated - no matching columns found")
    
    print(f"\nValidation completed!")

if __name__ == "__main__":
    validate_annotations() 