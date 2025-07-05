#
# run_finetune.py
#
# Now includes a final evaluation step to measure performance on the test set.
#

import os
import json
import re
import sys
import subprocess

# --- CONFIG ---
TRAIN_FILE = 'training_data.jsonl'
TEST_FILE = 'testing_data.jsonl'


def run_pipeline():
    """Main function to run the full pipeline: setup, tune, and evaluate."""
    # This will take a while and needs a GPU.
    tuned_model = setup_and_tune(TRAIN_FILE)

    # After training, evaluate the model on the held-out test data.
    evaluate_model(tuned_model, TEST_FILE)


def setup_and_tune(training_data_file):
    """Handles environment setup, model loading, and tuning."""
    print("--- Starting Model Fine-Tuning ---")
    # --- Part 1: Environment Setup ---
    print("Setting up environment...")
    os.environ["KERAS_BACKEND"] = "jax"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.00"

    # Added scikit-learn, matplotlib, and seaborn for evaluation.
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "keras", "keras-hub", "scikit-learn", "matplotlib", "seaborn"], check=True)
    except subprocess.CalledProcessError:
        print("Could not install packages. Check pip and internet connection.")
        sys.exit(1)

    # --- IMPORTANT: KAGGLE API KEYS ---
    # os.environ["KAGGLE_USERNAME"] = "your_username_here"
    # os.environ["KAGGLE_KEY"] = "your_key_here"
    if 'KAGGLE_USERNAME' not in os.environ or 'KAGGLE_KEY' not in os.environ:
        print("ERROR: Kaggle credentials not found.")
        sys.exit(1)

    import keras
    import keras_hub

    # --- Part 2: Load Model and Configure LoRA ---
    print("Loading Gemma 1b model...")
    gemma_lm = keras_hub.models.Gemma3CausalLM.from_preset("gemma3_instruct_1b")
    gemma_lm.backbone.enable_lora(rank=4)

    # --- Part 3: Load Data and Train ---
    print(f"Loading training data from {training_data_file}...")
    with open(training_data_file) as f:
        training_data = [json.loads(line) for line in f]
    
    fit_data = { "prompts": [item['instruction'] for item in training_data], "responses": [item['response'] for item in training_data] }
    
    gemma_lm.preprocessor.sequence_length = 512
    optimizer = keras.optimizers.AdamW(learning_rate=5e-5, weight_decay=0.01)
    
    gemma_lm.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=optimizer,
        weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )

    print(f"\nStarting training on {len(fit_data['prompts'])} examples...")
    gemma_lm.fit(fit_data, epochs=1, batch_size=1)

    print("\n--- Fine-Tuning Complete! ---")
    return gemma_lm


def evaluate_model(model, test_file):
    """Runs the model on the test set and prints evaluation metrics."""
    print(f"\n--- Evaluating model on {test_file} ---")
    from sklearn.metrics import classification_report, confusion_matrix, multilabel_confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    # Load the test data.
    with open(test_file) as f:
        test_data = [json.loads(line) for line in f]

    true_labels = []
    pred_labels = []

    # Get predictions for each item in the test set.
    for item in test_data:
        instruction = item['instruction']
        ground_truth_response = json.loads(item['response']) # The correct answer
        
        # --- Get ground truth labels ---
        # For this example, we'll just evaluate the implicit aspects.
        # Evaluating token-level explicit aspects is more complex.
        true_implicit_aspects = set(ground_truth_response.get('implicit_aspects', []))
        
        # --- Get model prediction ---
        prompt = f"Instruction:\n{instruction}\n\nResponse:\n"
        raw_output = model.generate(prompt, max_length=512)
        response_part = raw_output.split("Response:")[-1].strip()
        
        pred_implicit_aspects = set()
        try:
            json_match = re.search(r'\{.*\}', response_part, re.DOTALL)
            if json_match:
                parsed_json = json.loads(json_match.group(0))
                pred_implicit_aspects = set(parsed_json.get('implicit_aspects', []))
        except json.JSONDecodeError:
            pass # Keep predicted aspects as an empty set if JSON fails.

        true_labels.append(true_implicit_aspects)
        pred_labels.append(pred_implicit_aspects)

    # --- Calculate and Display Metrics for Implicit Aspects ---
    # First, get a list of all possible labels.
    all_possible_labels = sorted(list(set.union(*true_labels, *pred_labels)))
    
    # Binarize the labels for scikit-learn.
    # Each review will have a list of 0s and 1s corresponding to the all_possible_labels list.
    true_binarized = [[1 if label in s else 0 for label in all_possible_labels] for s in true_labels]
    pred_binarized = [[1 if label in s else 0 for label in all_possible_labels] for s in pred_labels]

    # 1. Classification Report (the text table)
    print("\n--- Classification Report (for Implicit Aspects) ---")
    report = classification_report(true_binarized, pred_binarized, target_names=all_possible_labels, zero_division=0)
    print(report)

    # 2. Confusion Matrix (the visualization)
    print("\n--- Generating Confusion Matrix ---")
    # For multi-label, we sum the confusion matrices of each label.
    cm = multilabel_confusion_matrix(true_binarized, pred_binarized, labels=range(len(all_possible_labels)))
    # For a simple view, we can sum them up to a single matrix
    cm_summed = cm.sum(axis=0)

    # To make it easier to read: TP, FN, FP, TN
    # We will create a simpler version focusing on "when a label was predicted, was it right?"
    # which is essentially what the classification report shows.
    # A per-class heatmap is more direct.
    
    report_data = classification_report(true_binarized, pred_binarized, target_names=all_possible_labels, zero_division=0, output_dict=True)
    df = pd.DataFrame(report_data).transpose()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[['precision', 'recall', 'f1-score']].iloc[:-3], annot=True, cmap="viridis")
    plt.title("F1-Score, Precision, and Recall for Implicit Aspects")
    plt.show()


if __name__ == '__main__':
    # Make sure required files exist.
    if not os.path.exists('001-050 _ Thesis Annotation Sheet - Trial.csv'):
        print("Error: The CSV file is missing. Please add it to the directory.")
        sys.exit(1)
        
    run_pipeline()