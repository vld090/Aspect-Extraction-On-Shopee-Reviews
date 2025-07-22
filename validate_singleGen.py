import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

generated_file = 'geminiAPI/annotated_test_data.csv'
gt_file = 'valid_gen.csv'

def calculate_metrics(generated_file, gt_file):
    # Load the generated and ground truth data
    generated_df = pd.read_csv(generated_file)
    gt_df = pd.read_csv(gt_file)

    # Ensure both DataFrames have the same length
    if len(generated_df) != len(gt_df):
        raise ValueError("Generated and ground truth files must have the same number of rows.")

    # Extract the 'General Aspect' column from both DataFrames
    y_pred = generated_df['General Aspect'].values
    y_true = gt_df['General Aspect'].values

    # Calculate accuracy: percentage of reviews where gemini api correctly predicted the general aspect
    accuracy = accuracy_score(y_true, y_pred)

    # Calculate precision, recall, and F1-score
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)

    return accuracy, precision, recall, f1

def validate_annotations(generated_file, gt_file):
    accuracy, precision, recall, f1 = calculate_metrics(generated_file, gt_file)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Print classification report
    gt_df = pd.read_csv(gt_file)
    y_true = gt_df['General Aspect'].values
    y_pred = pd.read_csv(generated_file)['General Aspect'].values
    print("\nClassification Report:\n", classification_report(y_true, y_pred))


if __name__ == "__main__":
    validate_annotations(generated_file, gt_file) 