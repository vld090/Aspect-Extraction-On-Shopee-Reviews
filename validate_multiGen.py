import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# File paths
PREDICTED_FILE = 'geminiAPI/annotated_test_data.csv'
GROUND_TRUTH_FILE = 'valid_multigen.csv'

# Aspects to validate
ASPECTS = ['product', 'delivery', 'price', 'service']

def validate_single_aspect(pred_df, gt_df, aspect):
    """Validate a single aspect column"""
    y_pred = pred_df[aspect].fillna('0').astype(str)
    y_true = gt_df[aspect].fillna('0').astype(str)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    print(f"\n=== {aspect.upper()} ASPECT ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    return {
        'aspect': aspect,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def calculate_exact_match_metrics(pred_df, gt_df, aspects):
    """Calculate exact set matching metrics"""
    correct_samples = 0
    total_samples = len(pred_df)
    
    # For precision, recall, F1 - treat each sample as binary (all correct vs not all correct)
    y_true_binary = []
    y_pred_binary = []
    
    for i in range(total_samples):
        # Check if all aspects match for this sample
        all_correct = True
        for aspect in aspects:
            pred_val = str(pred_df.loc[i, aspect]) if pd.notna(pred_df.loc[i, aspect]) else '0'
            true_val = str(gt_df.loc[i, aspect]) if pd.notna(gt_df.loc[i, aspect]) else '0'
            
            if pred_val != true_val:
                all_correct = False
                break
        
        if all_correct:
            correct_samples += 1
            
        # binary classification metrics (1 = all correct, 0 = not all correct)
        y_true_binary.append(1)  # Ground truth is always "all should be correct"
        y_pred_binary.append(1 if all_correct else 0)  # Prediction success
    
    # Calculate metrics
    exact_match_accuracy = correct_samples / total_samples
    
    # For precision, recall, F1 in exact matching context:
    # Precision: Of samples we predicted as "all correct", how many were actually all correct
    # Recall: Of samples that should be "all correct", how many did we predict as all correct
    # precision, recall, f1, _ = precision_recall_fscore_support(
    #     y_true_binary, y_pred_binary, average='binary', zero_division=0
    # )
    
    return exact_match_accuracy, correct_samples, total_samples
def validate_all_aspects():
    """Main validation function"""
    # Load data
    pred_df = pd.read_csv(PREDICTED_FILE)
    gt_df = pd.read_csv(GROUND_TRUTH_FILE)
    
    print(f"Predicted data shape: {pred_df.shape}")
    print(f"Ground truth data shape: {gt_df.shape}")
    
    # Validate each aspect
    results = []
    
    for aspect in ASPECTS:
        if aspect in pred_df.columns and aspect in gt_df.columns:
            result = validate_single_aspect(pred_df, gt_df, aspect)
            results.append(result)
            
        else:
            print(f"WARNING: '{aspect}' column not found in both files")
    
    # Combined metrics
    valid_aspects = [aspect for aspect in ASPECTS 
                    if aspect in pred_df.columns and aspect in gt_df.columns]
    
    if valid_aspects:
        combined_accuracy, correct_count, total_count = \
            calculate_exact_match_metrics(pred_df, gt_df, valid_aspects)
        
        print(f"\n{'='*50}")
        print("EXACT MATCH (ALL ASPECTS)")
        print(f"{'='*50}")
        print(f"Samples with ALL aspects correct: {correct_count}/{total_count}")
        print(f"Accuracy: {combined_accuracy:.4f}")
    
        # Save results
        results_df = pd.DataFrame(results)
        results_df.to_csv('validation_results.csv', index=False)
        print(f"\nResults saved to 'validation_results.csv'")

if __name__ == "__main__":
    validate_all_aspects()