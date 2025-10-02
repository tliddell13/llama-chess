"""
MLX LoRA Fine-tuning for Chess Move Prediction
Phase 1: Basic move prediction baseline

Prerequisites: Run prepare_chess_data.py first to create chess_data/ directory
"""

import subprocess
from pathlib import Path

# Configuration
MODEL = "meta-llama/Llama-3.2-3B"
DATA_DIR = "chess_data"
ADAPTER_PATH = "chess_adapters"


def test_model_before_training():
    """Test the model before training"""
    print("\n" + "="*60)
    print("TESTING MODEL BEFORE TRAINING")
    print("="*60)
    
    test_prompt = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 ->"
    
    print(f"Test position: {test_prompt}")
    print("Generating prediction...\n")
    
    # Run the inference from command line
    result = subprocess.run([
        "mlx_lm.generate",
        "--model", MODEL,
        "--prompt", test_prompt,
        "--max-tokens", "10"
    ], capture_output=True, text=True)
    
    print(result.stdout)
    print("="*60 + "\n")


def train_model():
    """Run MLX-LM LoRA training"""
    
    # Check if data directory exists
    if not Path(DATA_DIR).exists():
        print(f"\nERROR: {DATA_DIR}/ directory not found!")
        print("Please run 'python split_data.py' first")
        return False
    
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    print(f"Model: {MODEL}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Adapter path: {ADAPTER_PATH}")
    print("="*60 + "\n")
    
    # Build the training command
    cmd = [
        "mlx_lm.lora",
        "--model", MODEL,
        "--train",
        "--data", DATA_DIR,
        "--adapter-path", ADAPTER_PATH,
        "--loss-function", "chess_masked",
        "--iters", "2000",            # Good balance
        "--batch-size", "4",          # Try increasing from 2
        "--num-layers", "16",         # Good (covers most of model)
        "--learning-rate", "1e-4",    # Standard for LoRA
        "--save-every", "250",        # Save 8 checkpoints total
        "--steps-per-eval", "100",    # Check val loss 20 times
        "--val-batches", "25"         # How many batches for validation
    ]
    
    print("Running command:")
    print(" ".join(cmd))
    print()
    
    # Run training
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print("\n" + "="*60)
        print("TRAINING COMPLETE!")
        print("="*60)
        print(f"Adapters saved to: {ADAPTER_PATH}")
    else:
        print("\n" + "="*60)
        print("TRAINING FAILED")
        print("="*60)
        return False
    
    return True


def test_model_after_training():
    """Test the model after training with adapters"""
    print("\n" + "="*60)
    print("TESTING MODEL AFTER TRAINING")
    print("="*60)
    
    test_prompt = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 ->"
    
    print(f"Test position: {test_prompt}")
    print("Generating prediction with trained adapters...\n")
    
    result = subprocess.run([
        "mlx_lm.generate",
        "--model", MODEL,
        "--adapter-path", ADAPTER_PATH,
        "--prompt", test_prompt,
        "--max-tokens", "10"
    ], capture_output=True, text=True)
    
    print(result.stdout)
    print("="*60)


def main():
    """Main training pipeline"""
    print("="*60)
    print("CHESS MOVE PREDICTION - PHASE 1")
    print("="*60)
    print("This is a TEST run with 200 examples")
    print("="*60 + "\n")
    
    # 1. Test before training
    test_model_before_training()
    
    # 2. Train
    success = train_model()
    
    if not success:
        print("\nTraining failed. Check the error messages above.")
        return
    
    # 3. Test after training
    test_model_after_training()
    
    print("\n" + "="*60)
    print("PHASE 1 TEST COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()