# Gemma 3 Simple LoRA Fine-tuning Template
# Based on Google's official documentation: https://ai.google.dev/gemma/docs/core/lora_tuning
# Simple implementation following Google's exact approach

import os
import json
import keras
import keras_hub

# Setup Kaggle authentication (required for Gemma models)
# You need to:
# 1. Go to https://www.kaggle.com/settings/account
# 2. Create a new API token and download kaggle.json
# 3. Place kaggle.json in ~/.kaggle/ directory

# Set environment variables for Kaggle (replace with your actual credentials)
os.environ["KAGGLE_USERNAME"] = "your_kaggle_username"
os.environ["KAGGLE_KEY"] = "your_kaggle_api_key"

def setup_gemma_model():
    """Setup Gemma model following Google's documentation."""
    print("Loading Gemma model...")
    
    # Load Gemma model using Keras Hub (following Google's approach)
    gemma_lm = keras_hub.KerasHubModel.from_pretrained("gemma-1.1-1b-it")
    
    print("Model loaded successfully!")
    return gemma_lm

def load_training_data(data_path):
    """
    Load training data in the format shown in Google's documentation.
    
    Args:
        data_path: Path to your JSONL training data file
    """
    print(f"Loading training data from: {data_path}")
    
    prompts = []
    responses = []
    line_count = 0
    
    with open(data_path) as file:
        for line in file:
            if line_count >= 1000:  # Limit examples for faster execution
                break
                
            examples = json.loads(line)
            
            # Filter out examples with context, to keep it simple
            if examples["context"]:
                continue
                
            # Format data into prompts and response lists
            prompts.append(examples["instruction"])
            responses.append(examples["response"])
            
            line_count += 1
    
    data = {
        "prompts": prompts,
        "responses": responses
    }
    
    print(f"Loaded {len(prompts)} training examples")
    return data

def setup_lora_tuning(gemma_lm):
    """Setup LoRA tuning following Google's documentation."""
    print("Setting up LoRA tuning...")
    
    # Enable LoRA for the model and set the LoRA rank to 4
    # Following Google's recommendation: start with small rank (4, 8, 16)
    gemma_lm.backbone.enable_lora(rank=4)
    
    # Check model summary to see reduced trainable parameters
    print("Model summary after enabling LoRA:")
    gemma_lm.summary()
    
    # Configure fine-tuning settings
    # Limit input sequence length to control memory usage
    gemma_lm.preprocessor.sequence_length = 256
    
    # Use AdamW optimizer (common for transformer models)
    optimizer = keras.optimizers.AdamW(
        learning_rate=5e-5,
        weight_decay=0.01,
    )
    # Exclude layernorm and bias terms from decay
    optimizer.exclude_from_weight_decay(var_names=["bias", "scale"])
    
    # Compile the model
    gemma_lm.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=optimizer,
        weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    
    print("LoRA tuning setup completed!")

def train_model(gemma_lm, data):
    """Train the model using Keras fit() method."""
    print("Starting training...")
    
    # Run fine-tuning process
    # This can take several minutes depending on your resources
    history = gemma_lm.fit(data, epochs=1, batch_size=1)
    
    print("Training completed!")
    return history

def test_model(gemma_lm):
    """Test the fine-tuned model with example prompts."""
    print("Testing fine-tuned model...")
    
    # Template for instruction-following format
    template = """Instruction:
{instruction}

Response:
{response}"""
    
    # Test with Europe trip prompt
    prompt = template.format(
        instruction="What should I do on a trip to Europe?",
        response="",
    )
    
    # Setup sampler for generation
    sampler = keras_hub.samplers.TopKSampler(k=5, seed=2)
    gemma_lm.compile(sampler=sampler)
    
    # Generate response
    response = gemma_lm.generate(prompt, max_length=256)
    print("Europe trip response:")
    print(response)
    
    # Test with photosynthesis prompt
    prompt = template.format(
        instruction="Explain the process of photosynthesis in a way that a child could understand.",
        response="",
    )
    
    response = gemma_lm.generate(prompt, max_length=256)
    print("\nPhotosynthesis response:")
    print(response)

def main():
    """Main function following Google's documentation exactly."""
    
    try:
        # Step 1: Setup Gemma model
        gemma_lm = setup_gemma_model()
        
        # Step 2: Load training data
        # Replace with your actual data file path
        data_path = "your_training_data.jsonl"
        data = load_training_data(data_path)
        
        # Step 3: Setup LoRA tuning
        setup_lora_tuning(gemma_lm)
        
        # Step 4: Train the model
        history = train_model(gemma_lm, data)
        
        # Step 5: Test the model
        test_model(gemma_lm)
        
        print("Fine-tuning pipeline completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have:")
        print("1. Set up Kaggle authentication")
        print("2. Installed required packages: keras, keras_hub")
        print("3. Have your training data in JSONL format")

if __name__ == "__main__":
    main() 