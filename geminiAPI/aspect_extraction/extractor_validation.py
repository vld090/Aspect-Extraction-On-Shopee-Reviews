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

def save_results(results: dict, results_file: str):
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