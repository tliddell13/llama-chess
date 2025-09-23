"""
Prepare chess training data for MLX-LM
Splits data into train/valid/test sets
"""

from pathlib import Path
import json

# Configuration
INPUT_FILE = "chess_training.jsonl"
OUTPUT_DIR = "chess_data"

# For testing my pipeline: use only a subset
USE_SUBSET = False
SUBSET_SIZE = 0

# Split ratios
TRAIN_RATIO = 0.8
VALID_RATIO = 0.1
TEST_RATIO = 0.1

def prepare_data():
    """Prepare training data in MLX-LM format"""
    
    print("="*60)
    print("PREPARING CHESS TRAINING DATA")
    print("="*60)
    
    # Create output directory
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(exist_ok=True)
    
    # Read input data
    print(f"\nReading from: {INPUT_FILE}")
    examples = []
    
    with open(INPUT_FILE, 'r') as f:
        for i, line in enumerate(f):
            if USE_SUBSET and i >= SUBSET_SIZE:
                break
            examples.append(line)
    
    total = len(examples)
    print(f"Loaded {total} examples")
    
    # Calculate split sizes
    train_size = int(TRAIN_RATIO * total)
    valid_size = int(VALID_RATIO * total)
    
    # Split data
    train_examples = examples[:train_size]
    valid_examples = examples[train_size:train_size + valid_size]
    test_examples = examples[train_size + valid_size:]
    
    print(f"\nSplit breakdown:")
    print(f"  Train: {len(train_examples)} examples ({TRAIN_RATIO*100:.0f}%)")
    print(f"  Valid: {len(valid_examples)} examples ({VALID_RATIO*100:.0f}%)")
    print(f"  Test:  {len(test_examples)} examples ({TEST_RATIO*100:.0f}%)")
    
    # Write files
    print(f"\nWriting to {OUTPUT_DIR}/")
    
    with open(output_path / "train.jsonl", 'w') as f:
        for line in train_examples:
            f.write(line)
    print("  ✓ train.jsonl")
    
    with open(output_path / "valid.jsonl", 'w') as f:
        for line in valid_examples:
            f.write(line)
    print("  ✓ valid.jsonl")
    
    with open(output_path / "test.jsonl", 'w') as f:
        for line in test_examples:
            f.write(line)
    print("  ✓ test.jsonl")
    
    print("\n" + "="*60)
    print("DATA PREPARATION COMPLETE!")
    print("="*60)
    print(f"\nReady for training with: --data {OUTPUT_DIR}")
    
    # Show example
    print("\nExample from training set:")
    example = json.loads(train_examples[0])
    print(f"  {example['text']}")

if __name__ == "__main__":
    prepare_data()