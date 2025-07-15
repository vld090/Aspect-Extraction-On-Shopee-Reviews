import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import re

# === CONFIGURABLE FILE PATHS ===
GENERATED_FILE = 'geminiAPI/annotated_test_data.csv'
GROUND_TRUTH_FILE = 'validate-test-data.csv'

def parse_explicit_column(explicit_str):
    """
    Parse the Explicit column from validate-test-data.csv into a list of (token, BIO, aspect) tuples.
    """
    if pd.isna(explicit_str) or explicit_str.strip() == '':
        return []
    # Remove leading/trailing quotes and spaces
    explicit_str = explicit_str.strip().strip('"')
    # Split by comma, but handle quoted tokens
    items = re.findall(r"'([^']+)' \(([^)]*)\)", explicit_str)
    result = []
    for token, tag in items:
        tag = tag.strip()
        if tag == 'O -' or tag == 'O - ' or tag == 'O -':
            bio, aspect = 'O', ''
        elif tag.startswith('B -') or tag.startswith('I -'):
            parts = tag.split('-')
            bio = parts[0].strip()
            aspect = parts[1].strip() if len(parts) > 1 else ''
        else:
            bio, aspect = 'O', ''
        result.append((token, bio, aspect))
    return result

def parse_implicit_column(implicit_str):
    """
    Parse the Implicit column from validate-test-data.csv into a list of (token, aspect) tuples.
    """
    if pd.isna(implicit_str) or implicit_str.strip() == '':
        return []
    implicit_str = implicit_str.strip().strip('"')
    items = re.findall(r"'([^']+)' \(([^)]*)\)", implicit_str)
    result = []
    for token, aspect in items:
        aspect = aspect.strip()
        result.append((token, aspect))
    return result

# === DATA LOADING (now for two files of the same format) ===
def load_and_clean_data(generated_file, ground_truth_file):
    """
    Load and clean the annotation data from both files (same format: Review #, Review, Explicit, Implicit)
    """
    try:
        generated_df = pd.read_csv(generated_file)
        ground_truth_df = pd.read_csv(ground_truth_file)
        print(f"Generated shape: {generated_df.shape}, columns: {list(generated_df.columns)}")
        print(f"Ground truth shape: {ground_truth_df.shape}, columns: {list(ground_truth_df.columns)}")
        # Parse explicit/implicit columns for both
        for df in [generated_df, ground_truth_df]:
            df['ExplicitParsed'] = df['Explicit'].apply(parse_explicit_column)
            df['ImplicitParsed'] = df['Implicit'].apply(parse_implicit_column)
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

# === VALIDATION FUNCTIONS (compare by Review #, token order) ===
def validate_explicit_aspects(generated_df, ground_truth_df):
    print(f"\n{'='*60}")
    print(f"EXPLICIT ASPECTS VALIDATION (token-level)")
    print(f"{'='*60}")
    all_gt_bio, all_pred_bio = [], []
    all_gt_aspect, all_pred_aspect = [], []
    for idx, gt_row in ground_truth_df.iterrows():
        review_num = gt_row['Review #']
        gt_explicit = gt_row['ExplicitParsed']
        pred_row = generated_df[generated_df['Review #'] == review_num]
        if pred_row.empty:
            continue
        pred_explicit = pred_row.iloc[0]['ExplicitParsed']
        for i, (gt_token, gt_bio, gt_aspect) in enumerate(gt_explicit):
            if i < len(pred_explicit):
                _, pred_bio, pred_aspect = pred_explicit[i]
                all_gt_bio.append(gt_bio)
                all_pred_bio.append(pred_bio)
                all_gt_aspect.append(gt_aspect)
                all_pred_aspect.append(pred_aspect)
    print("\nBIO Tag Metrics:")
    bio_metrics = calculate_metrics(all_gt_bio, all_pred_bio, label='Explicit BIO Tag')
    print("\nAspect Tag Metrics:")
    aspect_metrics = calculate_metrics(all_gt_aspect, all_pred_aspect, label='Explicit Aspect Tag')
    return {'BIO Tag': bio_metrics, 'Aspect Tag': aspect_metrics}

def validate_implicit_aspects(generated_df, ground_truth_df):
    print(f"\n{'='*60}")
    print(f"IMPLICIT ASPECTS VALIDATION (token-level)")
    print(f"{'='*60}")
    all_gt_aspect, all_pred_aspect = [], []
    for idx, gt_row in ground_truth_df.iterrows():
        review_num = gt_row['Review #']
        gt_implicit = gt_row['ImplicitParsed']
        pred_row = generated_df[generated_df['Review #'] == review_num]
        if pred_row.empty:
            continue
        pred_implicit = pred_row.iloc[0]['ImplicitParsed']
        for i, (gt_token, gt_aspect) in enumerate(gt_implicit):
            if i < len(pred_implicit):
                _, pred_aspect = pred_implicit[i]
                all_gt_aspect.append(gt_aspect)
                all_pred_aspect.append(pred_aspect)
    aspect_metrics = calculate_metrics(all_gt_aspect, all_pred_aspect, label='Implicit Aspect Tag')
    return {'Aspect Tag': aspect_metrics}

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

# === MAIN ===
def validate_annotations():
    print("Starting annotation validation (token-level, same format files)...")
    generated_df, ground_truth_df = load_and_clean_data(GENERATED_FILE, GROUND_TRUTH_FILE)
    if generated_df is None or ground_truth_df is None:
        print("Failed to load data files")
        return
    explicit_metrics = validate_explicit_aspects(generated_df, ground_truth_df)
    implicit_metrics = validate_implicit_aspects(generated_df, ground_truth_df)
    all_results = calculate_overall_metrics(explicit_metrics, implicit_metrics)
    save_detailed_results(all_results)
    print(f"\nValidation completed!")

if __name__ == "__main__":
    validate_annotations() 