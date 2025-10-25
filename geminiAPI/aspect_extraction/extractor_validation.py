import json
from typing import List, Set
import os

def tokenize(phrase: str) -> Set[str]:
    """Convert a phrase into a set of tokens."""
    return set(phrase.lower().split())

def calculate_f1_score(predicted_tokens: Set[str], true_tokens: Set[str]) -> float:
    """Calculate F1 score given predicted and true token sets."""
    if not predicted_tokens and not true_tokens:
        return 1.0
    if not predicted_tokens or not true_tokens:
        return 0.0
    
    true_positives = len(predicted_tokens.intersection(true_tokens))
    precision = true_positives / len(predicted_tokens) if predicted_tokens else 0
    recall = true_positives / len(true_tokens) if true_tokens else 0
    
    if precision + recall == 0:
        return 0.0
    
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score

def calculate_review_f1_score(predicted_phrases: List[str], ground_truth_phrases: List[str]) -> float:
    """Calculate aggregated F1 score for all extractions in a single review."""
    # Combine all tokens from predicted phrases
    predicted_tokens = set()
    for phrase in predicted_phrases:
        predicted_tokens.update(tokenize(phrase))
    
    # Combine all tokens from ground truth phrases
    true_tokens = set()
    for phrase in ground_truth_phrases:
        true_tokens.update(tokenize(phrase))
    
    return calculate_f1_score(predicted_tokens, true_tokens)

def validate_extractions(predicted_file: str, ground_truth_file: str) -> dict:
    """Validate extractions and return metrics."""

    # Load files
    with open(predicted_file, 'r', encoding='utf-8') as f:
        predicted = json.load(f)
    with open(ground_truth_file, 'r', encoding='utf-8') as f:
        ground_truth = json.load(f)
    
    predicted = [p for p in predicted if p['review_no'] != "review no."]
    ground_truth = [g for g in ground_truth if g['review_no'] != "Review no."]

    total_f1 = 0
    results = {}
    
    # Create lookup dictionaries
    ground_truth_dict = {item['review_no']: item for item in ground_truth}
    predicted_dict = {item['review_no']: item for item in predicted}
    
    # Process all reviews from ground truth
    for review_no, truth_item in ground_truth_dict.items():
        true_phrases = [extraction['phrase'] for extraction in truth_item['extractions']]
        
        # If review exists in predictions, use its phrases; otherwise, empty list
        if review_no in predicted_dict:
            pred_phrases = [extraction['phrase'] for extraction in predicted_dict[review_no]['extractions']]
        else:
            pred_phrases = []  # Missing prediction counts as empty list
        
        # Calculate F1 score for this review
        f1_score = calculate_review_f1_score(pred_phrases, true_phrases)
        total_f1 += f1_score
        
        results[review_no] = {
            'f1_score': f1_score,
            'predicted_phrases': pred_phrases,
            'ground_truth_phrases': true_phrases,
            'is_missing': review_no not in predicted_dict
        }
    
    # Calculate average F1 score based on total number of ground truth reviews
    avg_f1 = total_f1 / len(ground_truth_dict)
    
    return {
        'average_f1': avg_f1,
        'review_results': results,
        'total_reviews': len(ground_truth_dict),
        'missing_reviews': len(ground_truth_dict) - len(predicted_dict)
    }

def save_f1_results(results: dict, results_file: str):
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write(f"Total reviews in ground truth: {results['total_reviews']}\n")
        f.write(f"Missing reviews: {results['missing_reviews']}\n")
        f.write(f"Average F1 Score: {results['average_f1']:.3f}\n\n")
        
        for review_no, review_results in results['review_results'].items():
            f.write(f"Review {review_no}:\n")
            f.write(f"F1 Score: {review_results['f1_score']:.3f}\n")
            if review_results['is_missing']:
                f.write("Status: Missing in predictions\n")
            f.write(f"Predicted: {review_results['predicted_phrases']}\n")
            f.write(f"Ground Truth: {review_results['ground_truth_phrases']}\n\n")
    
    print(f"Results have been saved to {results_file}")

def load_results_from_txt(results_file: str) -> dict:
    """Load results from a text file."""
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"{results_file} does not exist.")
    
    with open(results_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    results = {
        'total_reviews': int(lines[0].strip().split(': ')[1]),
        'missing_reviews': int(lines[1].strip().split(': ')[1]),
        'average_f1': float(lines[2].strip().split(': ')[1]),
        'review_results': {}
    }
    
    current_review = None
    for line in lines[4:]:
        line = line.strip()
        if line.startswith("Review"):
            current_review = line.split()[1].strip(':')
            results['review_results'][current_review] = {}
        elif line.startswith("F1 Score"):
            results['review_results'][current_review]['f1_score'] = float(line.split(': ')[1])
        elif line.startswith("Status"):
            results['review_results'][current_review]['is_missing'] = True
        elif line.startswith("Predicted"):
            preds = line.split(': ')[1].strip('[]').replace("'", "").split(', ')
            results['review_results'][current_review]['predicted_phrases'] = preds if preds != [''] else []
        elif line.startswith("Ground Truth"):
            truths = line.split(': ')[1].strip('[]').replace("'", "").split(', ')
            results['review_results'][current_review]['ground_truth_phrases'] = truths if truths != [''] else []
    
    return results

def extraction_error_analysis(results: dict) -> dict:
    """Analyze common extraction errors."""
    analysis = {
        'partial': [],
        'over': [],
        'under': [],
        'perfect': [],
        'incorrect': []
    }
    
    for review_no, result in results['review_results'].items():
        predicted_tokens = set()
        for phrase in result['predicted_phrases']:
            predicted_tokens.update(tokenize(phrase))
        
        true_tokens = set()
        for phrase in result['ground_truth_phrases']:
            true_tokens.update(tokenize(phrase))
        
        review_summary = {
            'review_no': review_no,
            'f1_score': result['f1_score'],
            'predicted': result['predicted_phrases'],
            'ground_truth': result['ground_truth_phrases']
        }

        # 1. Perfect Match (F1 = 1.0)
        if result['f1_score'] == 1.0:
            analysis['perfect'].append(review_summary)
            continue
        # 2. Token-set comparison
        if predicted_tokens.issubset(true_tokens) and predicted_tokens != true_tokens:
            # Under-Extraction: The prediction is shorter and less comprehensive than the truth
            analysis['under'].append(review_summary)
        
        elif true_tokens.issubset(predicted_tokens) and predicted_tokens != true_tokens:
            # Over-Extraction: The prediction contains all the truth tokens plus extra noise/redundancy
            analysis['over'].append(review_summary)
        
        elif predicted_tokens.intersection(true_tokens) and result['f1_score'] >= 0.25:
             # Mismatch/Partial: Both sets have unique tokens, indicating a partial match and some error on both sides.
             analysis['partial'].append(review_summary)

        elif result['f1_score'] < 0.25:
            # Incorrect Case: Very low F1 score with little to no overlap
            analysis['incorrect'].append(review_summary)
    
    return analysis

def save_error_analysis(analysis: dict, analysis_file: str):
    """Save error analysis results to a file."""
    with open(analysis_file, 'w', encoding='utf-8') as f:
        f.write("## Extraction Error Analysis\n\n")

        # Summary of error types
        f.write("### Summary by Error Type\n")
        f.write(f"Perfect Matches (F1=1.000): {len(analysis['perfect'])}\n")
        f.write(f"Under-Extraction Cases: {len(analysis['under'])}\n")
        f.write(f"Over-Extraction Cases: {len(analysis['over'])}\n")
        f.write(f"Partial Cases: {len(analysis['partial'])}\n")
        f.write(f"Incorrect Cases (F1<0.25): {len(analysis['incorrect'])}\n")
        f.write("\n" + "-"*50 + "\n\n")

        # Under-Extraction
        f.write("### 1. Under-Extraction Cases (Predicted tokens are a subset of Ground Truth)\n")
        if analysis['under']:
            for item in analysis['under']:
                f.write(f"Review {item['review_no']} (F1: {item['f1_score']:.3f}):\n")
                f.write(f"  Predicted: {item['predicted']}\n")
                f.write(f"  Ground Truth: {item['ground_truth']}\n\n")
        else:
            f.write("No pure Under-Extraction cases found.\n\n")

        # Over-Extraction
        f.write("### 2. Over-Extraction Cases (Ground Truth tokens are a subset of Predicted)\n")
        if analysis['over']:
            for item in analysis['over']:
                f.write(f"Review {item['review_no']} (F1: {item['f1_score']:.3f}):\n")
                f.write(f"  Predicted: {item['predicted']}\n")
                f.write(f"  Ground Truth: {item['ground_truth']}\n\n")
        else:
            f.write("No pure Over-Extraction cases found.\n\n")

        # Mismatch/Partial
        f.write("### 3. Partial Cases (Token sets have unique tokens in both)\n")
        if analysis['partial']:
            for item in analysis['partial']:
                f.write(f"Review {item['review_no']} (F1: {item['f1_score']:.3f}):\n")
                f.write(f"  Predicted: {item['predicted']}\n")
                f.write(f"  Ground Truth: {item['ground_truth']}\n\n")
        else:
            f.write("No Partial cases found.\n\n")

        # Incorrect Cases
        f.write("### 4. Incorrect Cases (F1 < 0.25 with little to no overlap)\n")
        if analysis['incorrect']:
            for item in analysis['incorrect']:
                f.write(f"Review {item['review_no']} (F1: {item['f1_score']:.3f}):\n")
                f.write(f"  Predicted: {item['predicted']}\n")
                f.write(f"  Ground Truth: {item['ground_truth']}\n\n")
        else:
            f.write("No Incorrect cases found.\n\n")

    print(f"Error analysis has been saved to {analysis_file}")

# if __name__ == "__main__":
#     predicted_file = "output.json"
#     ground_truth_file = "ground_truth.json"
#     results_file = "extractor_validation2_results.txt"
    
#     # First run the CSV to JSON conversion
#     from utils import csv_to_json
#     csv_to_json('new_extracted_data.csv', predicted_file)
    
#     # Then validate the extractions
#     results = validate_extractions(predicted_file, ground_truth_file)
    
#     # Save results to text file
#     with open(results_file, 'w', encoding='utf-8') as f:
#         f.write(f"Total reviews in ground truth: {results['total_reviews']}\n")
#         f.write(f"Missing reviews: {results['missing_reviews']}\n")
#         f.write(f"Average F1 Score: {results['average_f1']:.3f}\n\n")
        
#         for review_no, review_results in results['review_results'].items():
#             f.write(f"Review {review_no}:\n")
#             f.write(f"F1 Score: {review_results['f1_score']:.3f}\n")
#             if review_results['is_missing']:
#                 f.write("Status: Missing in predictions\n")
#             f.write(f"Predicted: {review_results['predicted_phrases']}\n")
#             f.write(f"Ground Truth: {review_results['ground_truth_phrases']}\n\n")
    
#     print(f"Results have been saved to {results_file}")