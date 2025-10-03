import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, hamming_loss

# File paths
PREDICTED_FILE = 'restructured_test.csv'
GROUND_TRUTH_FILE = 'valid_multigen.csv'

# Aspects to validate
ASPECTS = ['product', 'delivery', 'price', 'service']

def validate_single_aspect(pred_df, gt_df, aspect):
    """Validate a single aspect column"""
    y_pred = pred_df[aspect].fillna('0').astype(str)
    y_true = gt_df[aspect].fillna('0').astype(str)
    
    accuracy = accuracy_score(y_true, y_pred)
    
    print(f"\n=== {aspect.upper()} ASPECT ===")
    print(f"Accuracy: {accuracy:.4f}")
    
    return {
        'aspect': aspect,
        'accuracy': accuracy
    }

def calculate_exact_match_metrics(pred_df, gt_df, aspects):
    """Calculate exact set matching metrics and hamming loss"""
    correct_samples = 0
    total_samples = len(pred_df)
    
    # For precision, recall, F1 - treat each sample as binary (all correct vs not all correct)
    y_true_binary = []
    y_pred_binary = []
    
    # For hamming loss calculation
    y_true_matrix = []
    y_pred_matrix = []
    
    for i in range(total_samples):
        # Check if all aspects match for this sample
        all_correct = True
        sample_true = []
        sample_pred = []
        
        for aspect in aspects:
            pred_val = str(pred_df.loc[i, aspect]) if pd.notna(pred_df.loc[i, aspect]) else '0'
            true_val = str(gt_df.loc[i, aspect]) if pd.notna(gt_df.loc[i, aspect]) else '0'
            
            # Convert to binary for hamming loss
            sample_true.append(1 if true_val != '0' else 0)
            sample_pred.append(1 if pred_val != '0' else 0)
            
            if pred_val != true_val:
                all_correct = False
        
        if all_correct:
            correct_samples += 1
            
        # Add to matrices for hamming loss
        y_true_matrix.append(sample_true)
        y_pred_matrix.append(sample_pred)
        
        # binary classification metrics (1 = all correct, 0 = not all correct)
        y_true_binary.append(1)  # Ground truth is always "all should be correct"
        y_pred_binary.append(1 if all_correct else 0)  # Prediction success
    
    # Calculate metrics
    exact_match_accuracy = correct_samples / total_samples
    
    # Calculate hamming loss
    h_loss = hamming_loss(y_true_matrix, y_pred_matrix)
    
    return exact_match_accuracy, correct_samples, total_samples, h_loss

def validate_all_aspects():
    """Main validation function"""
    # Load data
    pred_df = pd.read_csv(PREDICTED_FILE)
    gt_df = pd.read_csv(GROUND_TRUTH_FILE)
    
    print(f"Predicted data shape: {pred_df.shape}")
    print(f"Ground truth data shape: {gt_df.shape}")
    
    # Store results for text file
    results_text = []
    results_text.append(f"Validation Results\n{'='*50}\n")
    results_text.append(f"Predicted file: {PREDICTED_FILE}")
    results_text.append(f"Ground truth file: {GROUND_TRUTH_FILE}\n")
    
    # Validate each aspect
    aspect_results = []
    
    for aspect in ASPECTS:
        if aspect in pred_df.columns and aspect in gt_df.columns:
            result = validate_single_aspect(pred_df, gt_df, aspect)
            aspect_results.append(result)
            results_text.append(f"\n{aspect.upper()} ASPECT")
            results_text.append(f"Accuracy: {result['accuracy']:.4f}")
        else:
            print(f"WARNING: '{aspect}' column not found in both files")
            results_text.append(f"\nWARNING: '{aspect}' column not found in both files")
    
    # Combined metrics
    valid_aspects = [aspect for aspect in ASPECTS 
                    if aspect in pred_df.columns and aspect in gt_df.columns]
    
    if valid_aspects:
        combined_accuracy, correct_count, total_count, hamming_loss_score = \
            calculate_exact_match_metrics(pred_df, gt_df, valid_aspects)
        
        results_text.append(f"\n{'='*50}")
        results_text.append("EXACT MATCH (ALL ASPECTS)")
        results_text.append(f"{'='*50}")
        results_text.append(f"Samples with ALL aspects correct: {correct_count}/{total_count}")
        results_text.append(f"Accuracy: {combined_accuracy:.4f}")
        results_text.append(f"Hamming Loss: {hamming_loss_score:.4f}")
    
        # Save results to text file
        with open('validation_results.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(results_text))
        print(f"\nResults saved to 'validation_results.txt'")

if __name__ == "__main__":
    validate_all_aspects()