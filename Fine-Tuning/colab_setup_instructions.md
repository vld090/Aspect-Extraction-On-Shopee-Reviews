# Fine-Tuning Setup Instructions for Google Colab

## Prerequisites

### 1. Kaggle API Setup
1. Go to [kaggle.com](https://kaggle.com) and create an account
2. Go to your **Account** page → **API** section
3. Click **Create New Token** to download `kaggle.json`
4. Open the downloaded file and note your username and API key

### 2. Colab Runtime Setup
1. Open [Google Colab](https://colab.research.google.com)
2. Create a new notebook
3. Go to **Runtime** → **Change runtime type**
4. Set **Hardware accelerator** to **T4 GPU** (or V100 if available)
5. Set **Runtime type** to **Python 3**

## Step-by-Step Instructions

### Step 1: Upload Your Data Files

1. **Upload the CSV file:**
   ```python
   from google.colab import files
   uploaded = files.upload()
   ```
   - Upload `001-050 _ Thesis Annotation Sheet - Trial.csv`

2. **Create the Python files:**
   - Copy the contents of `prepare_data.py` and `run_finetune.py` into separate cells
   - Or upload them directly using the file upload feature

### Step 2: Set Up Kaggle Credentials

```python
# Add this cell to set your Kaggle credentials
import os
from google.colab import userdata

# Set your Kaggle credentials (you'll be prompted to enter them)
os.environ["KAGGLE_USERNAME"] = "your_kaggle_username_here"
os.environ["KAGGLE_KEY"] = "your_kaggle_api_key_here"
```

**Alternative method using Colab Secrets:**
1. Click the **🔑** (Secrets) icon in the left sidebar
2. Add two secrets:
   - Name: `KAGGLE_USERNAME`, Value: your Kaggle username
   - Name: `KAGGLE_KEY`, Value: your Kaggle API key
3. Then use this code:
   ```python
   import os
   from google.colab import userdata
   
   os.environ["KAGGLE_USERNAME"] = userdata.get('KAGGLE_USERNAME')
   os.environ["KAGGLE_KEY"] = userdata.get('KAGGLE_KEY')
   ```

### Step 3: Install Required Packages

```python
# Install required packages
!pip install -q -U keras keras-hub scikit-learn matplotlib seaborn
```

### Step 4: Set Environment Variables

```python
# Set backend and memory configuration
import os
os.environ["KERAS_BACKEND"] = "jax"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.00"
```

### Step 5: Prepare Your Data

```python
# Copy the prepare_data.py content here
import csv
import json
import random
from collections import defaultdict

# --- CONFIG ---
INPUT_CSV = '001-050 _ Thesis Annotation Sheet - Trial.csv'
TRAIN_FILE_OUT = 'training_data.jsonl'
TEST_FILE_OUT = 'testing_data.jsonl'
TRAIN_SPLIT_RATIO = 0.8

def create_training_and_testing_data():
    """
    Processes the CSV and splits the data into two separate JSONL files.
    """
    print(f"Reading from {INPUT_CSV}...")
    reviews = defaultdict(list)
    with open(INPUT_CSV, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('Review #'):
                reviews[row['Review #']].append(row)

    # Convert defaultdict to a list and shuffle it to ensure random distribution.
    review_items = list(reviews.items())
    random.shuffle(review_items)

    # Calculate the split index.
    split_idx = int(len(review_items) * TRAIN_SPLIT_RATIO)
    train_reviews = review_items[:split_idx]
    test_reviews = review_items[split_idx:]

    print(f"Total reviews: {len(review_items)}. Splitting into {len(train_reviews)} training and {len(test_reviews)} testing examples.")

    # Process both training and testing sets.
    _write_jsonl_file(train_reviews, TRAIN_FILE_OUT)
    _write_jsonl_file(test_reviews, TEST_FILE_OUT)

    print(f"\nDone! Data successfully split into {TRAIN_FILE_OUT} and {TEST_FILE_OUT}.")

def _write_jsonl_file(review_set, output_filename):
    """Helper function to process and write a set of reviews to a JSONL file."""
    with open(output_filename, 'w', encoding='utf-8') as f:
        for review_id, rows in review_set:
            full_review_text = " ".join([row['Token'] for row in rows])

            explicit_aspects = []
            implicit_aspects = set()

            for row in rows:
                if row['BIO Tag (For Explicit Aspects)'] in ('B', 'I'):
                    tag = f"{row['BIO Tag (For Explicit Aspects)']} - {row['Aspect Tag (For Explicit Aspects)']}"
                    explicit_aspects.append({"token": row['Token'], "tag": tag})

                if row['Final Tag (For Implicit Aspects)']:
                    implicit_tag = row['Final Tag (For Implicit Aspects)'].strip().split(' - ')[-1]
                    if implicit_tag:
                        implicit_aspects.add(implicit_tag)

            instruction = (
                "Analyze the following customer review to identify all explicit and implicit "
                "product or service aspects. Return the answer in JSON format with two keys: "
                "'explicit_aspects' and 'implicit_aspects'.\n\n"
                f"Review: \"{full_review_text}\""
            )

            response_json = {
                "explicit_aspects": explicit_aspects,
                "implicit_aspects": sorted(list(implicit_aspects))
            }

            record = {
                "instruction": instruction,
                "response": json.dumps(response_json, indent=2)
            }
            f.write(json.dumps(record) + '\n')

# Run the data preparation
create_training_and_testing_data()
```

### Step 6: Run the Fine-Tuning

```python
# Copy the run_finetune.py content here
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

    # Check Kaggle credentials
    if 'KAGGLE_USERNAME' not in os.environ or 'KAGGLE_KEY' not in os.environ:
        print("ERROR: Kaggle credentials not found.")
        return None

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
    # Increased epochs and batch size for better training
    gemma_lm.fit(fit_data, epochs=3, batch_size=2)

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

# Run the pipeline
run_pipeline()
```

## Troubleshooting

### Common Issues:

1. **"Kaggle credentials not found"**
   - Make sure you've set the environment variables correctly
   - Check that your Kaggle API key is valid

2. **"CUDA out of memory"**
   - Reduce batch size from 2 to 1
   - Reduce sequence_length from 512 to 256
   - Try using a smaller model if available

3. **"Model download failed"**
   - Check your internet connection
   - Verify Kaggle credentials are correct
   - Try running the cell again

4. **"JSON decode error"**
   - This is expected for some predictions
   - The evaluation handles this gracefully

### Performance Tips:

1. **Use T4 GPU** for faster training
2. **Monitor GPU memory** in Colab's runtime info
3. **Save your model** after training:
   ```python
   # Save the fine-tuned model
   model.save_weights('fine_tuned_gemma_weights.h5')
   ```

### Expected Runtime:
- **Data preparation**: 1-2 minutes
- **Model loading**: 2-3 minutes
- **Fine-tuning**: 15-30 minutes (depending on data size)
- **Evaluation**: 5-10 minutes

## Next Steps

After successful fine-tuning:
1. Download the model weights
2. Test on new reviews
3. Analyze the results
4. Consider hyperparameter tuning if needed

## Notes

- The current setup uses only 1 epoch for quick testing
- For production use, consider 3-5 epochs
- The evaluation focuses on implicit aspects only
- Explicit aspect evaluation requires more complex token-level analysis 