import pandas as pd
import numpy as np
import os
from sklearn.metrics import accuracy_score, hamming_loss, f1_score

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
    
    return exact_match_accuracy, correct_samples, total_samples, h_loss, y_pred_matrix, y_true_matrix

def get_true_pred_aspects(pred_df: pd.DataFrame, gt_df: pd.DataFrame, aspect: str) -> list:
    result = []
    has_text = 'Review' in gt_df.columns

    for i in range(len(pred_df)):
        pred_val = str(pred_df.loc[i, aspect]).strip().lower() if pd.notna(pred_df.loc[i, aspect]) else '0'
        true_val = str(gt_df.loc[i, aspect]).strip().lower() if pd.notna(gt_df.loc[i, aspect]) else '0'
        
        predicted_binary = 1 if pred_val != '0' else 0
        actual_binary = 1 if true_val != '0' else 0

        sample_data = {
            'predicted': predicted_binary,
            'actual': actual_binary,
            'predicted_value': pred_val,
            'actual_value': true_val,
            'index': i
        }
        
        if has_text:
            # 'Review' from gt_df
            sample_data['Review'] = str(gt_df.loc[i, 'Review'])
            
        result.append(sample_data)
        
    return result

def identification_error_analysis(pred_df: pd.DataFrame, gt_df: pd.DataFrame, aspects: list) -> dict:
    """Analyze common identification errors for all aspects."""
    analysis = {
        'aspect': {}
    }

    for aspect in aspects:
        if aspect not in pred_df.columns or aspect not in gt_df.columns:
            continue
            
        results = get_true_pred_aspects(pred_df, gt_df, aspect)
        
        fp = [r for r in results if r['predicted'] == 1 and r['actual'] == 0] # False Positives (FP): Predicted 1, Actual 0 (Aspect *wrongly* identified)
        fn = [r for r in results if r['predicted'] == 0 and r['actual'] == 1] # False Negatives (FN): Predicted 0, Actual 1 (Aspect *missed*)
        
        tp = [r for r in results if r['predicted'] == 1 and r['actual'] == 1] # True Positives (TP): Predicted 1, Actual 1
        tn = [r for r in results if r['predicted'] == 0 and r['actual'] == 0] # True Negatives (TN): Predicted 0, Actual 0

        precision = len(tp) / (len(tp) + len(fp)) if (len(tp) + len(fp)) > 0 else 0.0  # Precision = TP / (TP + FP)
        recall = len(tp) / (len(tp) + len(fn)) if (len(tp) + len(fn)) > 0 else 0.0 # Recall = TP / (TP + FN)
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0 # F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
        
        analysis['aspect'][aspect] = {
            'true_positives': len(tp),
            'true_negatives': len(tn),
            'false_positives': len(fp),
            'false_negatives': len(fn),
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'fp_examples': fp[:5],  # Top 5 examples
            'fn_examples': fn[:5]   # Top 5 examples
        }
    
    return analysis

def save_error_analysis(analysis: dict, analysis_file: str):
    """Save error analysis results to a file."""
    results_text = ["Error Analysis: Aspect Identification\n" + "="*50 + "\n"]
    
    for aspect, data in analysis['aspect'].items():
        results_text.append(f"\n--- {aspect.upper()} ASPECT ---\n")
        results_text.append(f"Precision: {data['precision']:.4f}")
        results_text.append(f"Recall: {data['recall']:.4f}")
        results_text.append(f"F1: {data['f1_score']:.4f}")
        results_text.append(f"True Positives (TP): {data['true_positives']}")
        results_text.append(f"False Positives (FP - Aspect *wrongly* identified): {data['false_positives']}")
        results_text.append(f"False Negatives (FN - Aspect *missed*): {data['false_negatives']}")
        results_text.append(f"True Negatives (TN): {data['true_negatives']}")

        # FP Examples
        results_text.append("\nTOP 5 FALSE POSITIVE EXAMPLES (Model identified, but Ground Truth said '0'):")
        for i, fp_ex in enumerate(data['fp_examples']):
            text = fp_ex.get('Review', f"[Review text not available, index: {fp_ex['index']}]")
            results_text.append(f"  {i+1}. Pred Val: '{fp_ex['predicted_value']}'. Text: \"{text[:100]}...\"")
        
        # FN Examples
        results_text.append("\nTOP 5 FALSE NEGATIVE EXAMPLES (Model missed, but Ground Truth said *a value*):")
        for i, fn_ex in enumerate(data['fn_examples']):
            text = fn_ex.get('Review', f"[Review text not available, index: {fn_ex['index']}]")
            results_text.append(f"  {i+1}. Actual Val: '{fn_ex['actual_value']}'. Text: \"{text[:100]}...\"")
    
    # Save results to text file
    with open(analysis_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(results_text))
    print(f"\nError analysis has been saved to {analysis_file}")

def save_result_txt(results: dict, results_file: str):
        # Save results to text file
        with open(results_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(results['results_text']))
        print(f"\nResults saved to {results_file}")

def validate_all_aspects(predicted_file: str, ground_truth_file: str, aspects: list,
                         results_file: str, error_analysis_file: str) -> dict:
    """Main validation function"""
    # Load data
    pred_df = pd.read_csv(predicted_file)
    gt_df = pd.read_csv(ground_truth_file)
    
    print(f"Predicted data shape: {pred_df.shape}")
    print(f"Ground truth data shape: {gt_df.shape}")
    
    # Check if dataframes have the same length before proceeding
    if len(pred_df) != len(gt_df):
        print("ERROR: Predicted and Ground Truth files have different number of rows.")
        return {}
    
    # Store results for text file
    results_text = []
    results_text.append(f"Validation Results\n{'='*50}\n")
    
    # Validate each aspect
    aspect_results = []
    
    for aspect in aspects:
        if aspect in pred_df.columns and aspect in gt_df.columns:
            result = validate_single_aspect(pred_df, gt_df, aspect)
            aspect_results.append(result)
            results_text.append(f"\n{aspect.upper()} ASPECT")
            results_text.append(f"Accuracy: {result['accuracy']:.4f}")
        else:
            print(f"WARNING: '{aspect}' column not found in both files")
            results_text.append(f"\nWARNING: '{aspect}' column not found in both files")
    
    # Combined metrics
    valid_aspects = [aspect for aspect in aspects 
                    if aspect in pred_df.columns and aspect in gt_df.columns]
    
    if valid_aspects:
        combined_accuracy, correct_count, total_count, hamming_loss_score, y_true_matrix, y_pred_matrix = \
            calculate_exact_match_metrics(pred_df, gt_df, valid_aspects)
        
        if y_true_matrix:
        # Calculate micro and macro F1 scores
            micro_f1 = f1_score(y_true_matrix, y_pred_matrix, average='micro')
            macro_f1 = f1_score(y_true_matrix, y_pred_matrix, average='macro')

        results_text.append(f"\n{'='*50}")
        results_text.append("EXACT MATCH (ALL ASPECTS)")
        results_text.append(f"{'='*50}")
        results_text.append(f"Samples with ALL aspects correct: {correct_count}/{total_count}")
        results_text.append(f"Accuracy: {combined_accuracy:.4f}")
        results_text.append(f"Hamming Loss: {hamming_loss_score:.4f}")
        results_text.append(f"Micro F1 Score (Multi-Aspect): {micro_f1:.4f}")
        results_text.append(f"Macro F1 Score (Multi-Aspect): {macro_f1:.4f}")
    
    save_result_txt({'results_text': results_text}, results_file)

    # --- Error Analysis ---
    if valid_aspects:
        error_analysis_results = identification_error_analysis(pred_df, gt_df, valid_aspects)
        save_error_analysis(error_analysis_results, error_analysis_file)

    return {
        'results_text': results_text,
        'aspect_results': aspect_results,
        'combined_accuracy': combined_accuracy,
        'correct_count': correct_count,
        'total_count': total_count,
        'hamming_loss': hamming_loss_score,
        'micro_f1': micro_f1,
        'macro_f1': macro_f1
    }

def calculate_overall_performance(general_aspect_mapping: dict, error_analysis_files: dict) -> dict:
    """Calculate overall performance metrics for each general aspect group.
    
    Args:
        general_aspect_mapping: Dictionary mapping general aspects to their specific aspects
        error_analysis_files: Dictionary mapping general aspects to their error analysis file paths
    
    Returns:
        Dictionary containing aggregated metrics for each general aspect
    """
    overall_results = {}
    
    for general_aspect, specific_aspects in general_aspect_mapping.items():
        specific_aspects = [aspect.lower() for aspect in specific_aspects]
        # Initialize counters for this general aspect
        total_tp = 0
        total_fp = 0
        total_tn = 0
        total_fn = 0
        
        # Load error analysis file for this general aspect
        error_analysis_file = error_analysis_files[general_aspect]
        with open(error_analysis_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        # Process each specific aspect's metrics
        current_aspect = None
        for line in lines:
            line = line.strip()
            if line.startswith('---') and line.endswith('---') and 'ASPECT' in line:
                # Extract aspect name and clean it, removing 'ASPECT' and dashes
                current_aspect = line.replace('-', '').replace('ASPECT', '').strip().lower()
                continue
                
            if current_aspect in specific_aspects:
                if 'True Positives (TP):' in line:
                    total_tp += int(line.split(':')[1].strip())
                elif 'False Positives (FP' in line:
                    total_fp += int(line.split(':')[1].strip())
                elif 'False Negatives (FN' in line:
                    total_fn += int(line.split(':')[1].strip())
                elif 'True Negatives (TN):' in line:
                    total_tn += int(line.split(':')[1].strip())
        
        # Calculate overall metrics for this general aspect
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn) if (total_tp + total_tn + total_fp + total_fn) > 0 else 0.0
        
        overall_results[general_aspect] = {
            'true_positives': total_tp,
            'false_positives': total_fp,
            'true_negatives': total_tn,
            'false_negatives': total_fn,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'accuracy': accuracy,
            'specific_aspects': specific_aspects
        }
    
    return overall_results

def save_overall_results(results: dict, output_file: str):
    """Save overall performance results to a file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("Overall Performance by General Aspect\n")
        f.write("=" * 50 + "\n\n")
        
        for general_aspect, metrics in results.items():
            f.write(f"=== {general_aspect.upper()} ===\n")
            f.write(f"Specific aspects included: {', '.join(metrics['specific_aspects'])}\n\n")
            f.write(f"Aggregated Metrics:\n")
            f.write(f"True Positives (TP): {metrics['true_positives']}\n")
            f.write(f"False Positives (FP): {metrics['false_positives']}\n")
            f.write(f"True Negatives (TN): {metrics['true_negatives']}\n")
            f.write(f"False Negatives (FN): {metrics['false_negatives']}\n")
            f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"Precision: {metrics['precision']:.4f}\n")
            f.write(f"Recall: {metrics['recall']:.4f}\n")
            f.write(f"F1 Score: {metrics['f1_score']:.4f}\n\n")
    
    print(f"Overall results saved to {output_file}")

# Example usage:
# general_aspect_mapping = {
#     'price': ['price_value', 'price_comparison', 'price_discount'],
#     'quality': ['quality_material', 'quality_durability', 'quality_defects']
# }
# 
# error_analysis_files = {
#     'price': 'results/price_error_analysis.txt',
#     'quality': 'results/quality_error_analysis.txt'
# }
# 
# results = calculate_overall_performance(general_aspect_mapping, error_analysis_files)
# save_overall_results(results, 'results/overall_performance.txt')