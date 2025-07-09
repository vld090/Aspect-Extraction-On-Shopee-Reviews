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

def validate_explicit_aspects(generated_df, ground_truth_df):
    """
    Validate explicit aspects separately
    """
    print(f"\n{'='*60}")
    print(f"EXPLICIT ASPECTS VALIDATION")
    print(f"{'='*60}")
    
    explicit_columns = [
        ('BIO Tag (For Explicit Aspects)', 'BIO Tag (For Explicit Aspects)'),
        ('Aspect Tag (For Explicit Aspects)', 'Aspect Tag (For Explicit Aspects)'),
        ('General Aspect, Specific Aspect Category (For Explicit Aspects)', 'General Aspect, Specific Aspect Category (For Explicit Aspects)')
    ]
    
    explicit_metrics = {}
    
    for gen_col, gt_col in explicit_columns:
        if gen_col in generated_df.columns and gt_col in ground_truth_df.columns:
            print(f"\n{'='*50}")
            print(f"Comparing Explicit: {gen_col}")
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
            metrics = calculate_metrics(gt_tags, gen_tags, f"Explicit - {gen_col}")
            explicit_metrics[gen_col] = metrics
            
            # Show some examples
            print(f"\nSample comparisons (first 10):")
            for i in range(min(10, len(gen_tags))):
                print(f"  {i+1}: Ground Truth: '{gt_tags[i]}' | Generated: '{gen_tags[i]}' | Match: {gt_tags[i] == gen_tags[i]}")
        else:
            print(f"\nSkipping {gen_col} - column not found in one or both files")
    
    return explicit_metrics

def validate_implicit_aspects(generated_df, ground_truth_df):
    """
    Validate implicit aspects separately
    """
    print(f"\n{'='*60}")
    print(f"IMPLICIT ASPECTS VALIDATION")
    print(f"{'='*60}")
    
    implicit_columns = [
        ('Aspect Tag (For Implicit Aspects)', 'Aspect Tag (For Implicit Aspects)'),
        ('General Aspect, Specific Aspect Category (For Implicit Aspects)', 'General Aspect, Specific Aspect Category (For Implicit Aspects)')
    ]
    
    implicit_metrics = {}
    
    for gen_col, gt_col in implicit_columns:
        if gen_col in generated_df.columns and gt_col in ground_truth_df.columns:
            print(f"\n{'='*50}")
            print(f"Comparing Implicit: {gen_col}")
            print(f"{'='*50}")
            
            # Extract tags
            gen_tags = extract_aspect_tags(generated_df, gen_col)
            gt_tags = extract_aspect_tags(ground_truth_df, gt_col)
            
            # Ensure same length
            min_len = min(len(gen_tags), len(gt_tags))
            gen_tags = gen_tags[:min_len]
            gt_tags = gt_tags[:min_len]
            
            # Calculate metrics
            metrics = calculate_metrics(gt_tags, gen_tags, f"Implicit - {gen_col}")
            implicit_metrics[gen_col] = metrics
            
            # Show some examples
            print(f"\nSample comparisons (first 10):")
            for i in range(min(10, len(gen_tags))):
                print(f"  {i+1}: Ground Truth: '{gt_tags[i]}' | Generated: '{gen_tags[i]}' | Match: {gt_tags[i] == gen_tags[i]}")
        else:
            print(f"\nSkipping {gen_col} - column not found in one or both files")
    
    return implicit_metrics

def calculate_overall_metrics(explicit_metrics, implicit_metrics):
    """
    Calculate overall metrics for explicit and implicit aspects separately
    """
    print(f"\n{'='*80}")
    print(f"OVERALL SUMMARY METRICS")
    print(f"{'='*80}")
    
    # Explicit aspects summary
    if explicit_metrics:
        exp_avg_accuracy = np.mean([m['accuracy'] for m in explicit_metrics.values()])
        exp_avg_precision = np.mean([m['precision'] for m in explicit_metrics.values()])
        exp_avg_recall = np.mean([m['recall'] for m in explicit_metrics.values()])
        exp_avg_f1 = np.mean([m['f1'] for m in explicit_metrics.values()])
        
        print(f"\nEXPLICIT ASPECTS AVERAGE METRICS:")
        print(f"Average Accuracy: {exp_avg_accuracy:.4f}")
        print(f"Average Precision: {exp_avg_precision:.4f}")
        print(f"Average Recall: {exp_avg_recall:.4f}")
        print(f"Average F1 Score: {exp_avg_f1:.4f}")
    else:
        print(f"\nEXPLICIT ASPECTS: No metrics available")
    
    # Implicit aspects summary
    if implicit_metrics:
        imp_avg_accuracy = np.mean([m['accuracy'] for m in implicit_metrics.values()])
        imp_avg_precision = np.mean([m['precision'] for m in implicit_metrics.values()])
        imp_avg_recall = np.mean([m['recall'] for m in implicit_metrics.values()])
        imp_avg_f1 = np.mean([m['f1'] for m in implicit_metrics.values()])
        
        print(f"\nIMPLICIT ASPECTS AVERAGE METRICS:")
        print(f"Average Accuracy: {imp_avg_accuracy:.4f}")
        print(f"Average Precision: {imp_avg_precision:.4f}")
        print(f"Average Recall: {imp_avg_recall:.4f}")
        print(f"Average F1 Score: {imp_avg_f1:.4f}")
    else:
        print(f"\nIMPLICIT ASPECTS: No metrics available")
    
    # Combined overall metrics
    all_metrics = {**explicit_metrics, **implicit_metrics}
    if all_metrics:
        overall_avg_accuracy = np.mean([m['accuracy'] for m in all_metrics.values()])
        overall_avg_precision = np.mean([m['precision'] for m in all_metrics.values()])
        overall_avg_recall = np.mean([m['recall'] for m in all_metrics.values()])
        overall_avg_f1 = np.mean([m['f1'] for m in all_metrics.values()])
        
        print(f"\nCOMBINED OVERALL AVERAGE METRICS:")
        print(f"Average Accuracy: {overall_avg_accuracy:.4f}")
        print(f"Average Precision: {overall_avg_precision:.4f}")
        print(f"Average Recall: {overall_avg_recall:.4f}")
        print(f"Average F1 Score: {overall_avg_f1:.4f}")
    
    return {
        'explicit': explicit_metrics,
        'implicit': implicit_metrics,
        'combined': all_metrics
    }

def save_detailed_results(all_results):
    """
    Save detailed results to separate CSV files
    """
    # Save explicit results
    if all_results['explicit']:
        explicit_df = pd.DataFrame(all_results['explicit']).T
        explicit_df.to_csv('validation_results_explicit.csv')
        print(f"\nExplicit aspects results saved to 'validation_results_explicit.csv'")
    
    # Save implicit results
    if all_results['implicit']:
        implicit_df = pd.DataFrame(all_results['implicit']).T
        implicit_df.to_csv('validation_results_implicit.csv')
        print(f"Implicit aspects results saved to 'validation_results_implicit.csv'")
    
    # Save combined results
    if all_results['combined']:
        combined_df = pd.DataFrame(all_results['combined']).T
        combined_df.to_csv('validation_results_combined.csv')
        print(f"Combined results saved to 'validation_results_combined.csv'")

def validate_annotations():
    """
    Main function to validate annotations with separate explicit and implicit validation
    """
    print("Starting annotation validation with separate explicit and implicit analysis...")
    
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
    
    # Validate explicit aspects
    explicit_metrics = validate_explicit_aspects(generated_df, ground_truth_df)
    
    # Validate implicit aspects
    implicit_metrics = validate_implicit_aspects(generated_df, ground_truth_df)
    
    # Calculate overall metrics
    all_results = calculate_overall_metrics(explicit_metrics, implicit_metrics)
    
    # Save detailed results
    save_detailed_results(all_results)
    
    print(f"\nValidation completed with separate explicit and implicit analysis!")

if __name__ == "__main__":
    validate_annotations() 