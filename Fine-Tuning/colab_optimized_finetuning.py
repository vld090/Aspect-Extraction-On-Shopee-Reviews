# ============================================================================
# Model Fine-Tuning (Colab Optimized)
# ============================================================================

import keras
import keras_hub
import gc
import os

def setup_and_tune(training_data_file):
    """Handles environment setup, model loading, and tuning with memory optimization."""
    print("--- Starting Model Fine-Tuning (Optimized for Colab) ---")
    
    # Check Kaggle credentials
    if 'KAGGLE_USERNAME' not in os.environ or 'KAGGLE_KEY' not in os.environ:
        print("ERROR: Kaggle credentials not found.")
        return None

    # Memory optimization settings
    print("Setting up memory optimization...")
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"  # Use 80% of GPU memory
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"  # Allow GPU memory growth
    
    # Clear GPU memory
    try:
        import tensorflow as tf
        tf.keras.backend.clear_session()
        print("✓ Cleared TensorFlow session")
    except:
        pass
    
    # Load model with memory optimization
    print("Loading Gemma 1b model...")
    try:
        gemma_lm = keras_hub.models.Gemma3CausalLM.from_preset("gemma3_instruct_1b")
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return None

    # Configure LoRA with smaller rank to save memory
    print("Configuring LoRA...")
    gemma_lm.backbone.enable_lora(rank=2)  # Reduced from 4 to 2
    print("✓ LoRA configured")

    # Load training data
    print(f"Loading training data from {training_data_file}...")
    try:
        with open(training_data_file) as f:
            training_data = [json.loads(line) for line in f]
        print(f"✓ Loaded {len(training_data)} training examples")
    except Exception as e:
        print(f"✗ Error loading training data: {e}")
        return None
    
    fit_data = { 
        "prompts": [item['instruction'] for item in training_data], 
        "responses": [item['response'] for item in training_data] 
    }
    
    # Optimized model configuration
    print("Configuring model parameters...")
    gemma_lm.preprocessor.sequence_length = 256  # Reduced from 512 to 256
    optimizer = keras.optimizers.AdamW(
        learning_rate=3e-5,  # Slightly reduced learning rate
        weight_decay=0.01
    )
    
    gemma_lm.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=optimizer,
        weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    print("✓ Model compiled")

    # Training with memory management
    print(f"\nStarting training on {len(fit_data['prompts'])} examples...")
    print("Training parameters:")
    print(f"  - Epochs: 2 (reduced from 3)")
    print(f"  - Batch size: 1 (reduced from 2)")
    print(f"  - Sequence length: 256 (reduced from 512)")
    print(f"  - LoRA rank: 2 (reduced from 4)")
    
    try:
        # Train with smaller batch size and fewer epochs
        history = gemma_lm.fit(
            fit_data, 
            epochs=2,  # Reduced from 3
            batch_size=1,  # Reduced from 2
            verbose=1
        )
        print("✓ Training completed successfully")
        
        # Clear memory after training
        gc.collect()
        print("✓ Memory cleared")
        
    except Exception as e:
        print(f"✗ Training error: {e}")
        print("Try reducing batch_size to 1 or sequence_length to 128")
        return None

    print("\n--- Fine-Tuning Complete! ---")
    return gemma_lm

def setup_and_tune_conservative(training_data_file):
    """Even more conservative version for very limited memory."""
    print("--- Starting Model Fine-Tuning (Conservative Mode) ---")
    
    # Check Kaggle credentials
    if 'KAGGLE_USERNAME' not in os.environ or 'KAGGLE_KEY' not in os.environ:
        print("ERROR: Kaggle credentials not found.")
        return None

    # Very conservative memory settings
    print("Setting up conservative memory optimization...")
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.6"  # Use only 60% of GPU memory
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    
    # Clear GPU memory
    try:
        import tensorflow as tf
        tf.keras.backend.clear_session()
        print("✓ Cleared TensorFlow session")
    except:
        pass
    
    # Load model
    print("Loading Gemma 1b model...")
    try:
        gemma_lm = keras_hub.models.Gemma3CausalLM.from_preset("gemma3_instruct_1b")
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return None

    # Minimal LoRA configuration
    print("Configuring minimal LoRA...")
    gemma_lm.backbone.enable_lora(rank=1)  # Minimal rank
    print("✓ LoRA configured")

    # Load training data
    print(f"Loading training data from {training_data_file}...")
    try:
        with open(training_data_file) as f:
            training_data = [json.loads(line) for line in f]
        print(f"✓ Loaded {len(training_data)} training examples")
    except Exception as e:
        print(f"✗ Error loading training data: {e}")
        return None
    
    fit_data = { 
        "prompts": [item['instruction'] for item in training_data], 
        "responses": [item['response'] for item in training_data] 
    }
    
    # Very conservative model configuration
    print("Configuring conservative model parameters...")
    gemma_lm.preprocessor.sequence_length = 128  # Very short sequences
    optimizer = keras.optimizers.AdamW(
        learning_rate=1e-5,  # Very low learning rate
        weight_decay=0.01
    )
    
    gemma_lm.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=optimizer,
        weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    print("✓ Model compiled")

    # Training with minimal resources
    print(f"\nStarting training on {len(fit_data['prompts'])} examples...")
    print("Conservative training parameters:")
    print(f"  - Epochs: 1 (minimal)")
    print(f"  - Batch size: 1 (minimal)")
    print(f"  - Sequence length: 128 (very short)")
    print(f"  - LoRA rank: 1 (minimal)")
    
    try:
        # Train with minimal settings
        history = gemma_lm.fit(
            fit_data, 
            epochs=1,  # Just 1 epoch
            batch_size=1,  # Minimal batch size
            verbose=1
        )
        print("✓ Training completed successfully")
        
        # Clear memory
        gc.collect()
        print("✓ Memory cleared")
        
    except Exception as e:
        print(f"✗ Training error: {e}")
        print("Colab may not have enough GPU memory for this model.")
        print("Try using a smaller model or different runtime.")
        return None

    print("\n--- Fine-Tuning Complete! ---")
    return gemma_lm

# Choose your training mode:
# Option 1: Optimized (recommended for T4 GPU)
tuned_model = setup_and_tune('training_data.jsonl')

# Option 2: Conservative (if Option 1 crashes)
# tuned_model = setup_and_tune_conservative('training_data.jsonl')

# Option 3: If both crash, try this minimal version
def minimal_training_test():
    """Minimal test to check if training works at all."""
    print("Testing minimal training setup...")
    
    # Check if we can even load the model
    try:
        gemma_lm = keras_hub.models.Gemma3CausalLM.from_preset("gemma3_instruct_1b")
        print("✓ Model loads successfully")
        
        # Test with just 1 example
        test_data = {
            "prompts": ["Test instruction"],
            "responses": ["Test response"]
        }
        
        gemma_lm.preprocessor.sequence_length = 64
        gemma_lm.backbone.enable_lora(rank=1)
        
        gemma_lm.compile(
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=keras.optimizers.AdamW(learning_rate=1e-5),
        )
        
        # Try 1 step of training
        gemma_lm.fit(test_data, epochs=1, batch_size=1, steps_per_epoch=1)
        print("✓ Minimal training test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Minimal test failed: {e}")
        return False

# Uncomment to test minimal training:
# minimal_training_test() 