"""
Fuse LoRA adapters into base model to create standalone chess bot
"""
from pathlib import Path

def fuse_chess_model(base_model_path, adapter_path, output_path):
    """Fuse LoRA adapters with base model"""
    
    print("🔧 Fusing LoRA adapters with base model...")
    print(f"Base model: {base_model_path}")
    print(f"Adapters: {adapter_path}")
    print(f"Output: {output_path}")
    
    # Use subprocess to call the CLI command
    import subprocess
    
    cmd = [
        "python", "-m", "mlx_lm.fuse",
        "--model", base_model_path,
        "--adapter-path", adapter_path,
        "--save-path", output_path,
        "--de-quantize"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Model fusion complete!")
        print(f"Your standalone chess bot is ready at: {output_path}")
        print(f"\nTo use it:")
        print(f'model, tokenizer = load("{output_path}")')
    else:
        print("❌ Fusion failed!")
        print(f"Error: {result.stderr}")
        return False
    
    return True

def test_fused_model(model_path):
    """Test the fused model"""
    from mlx_lm import load, generate
    
    print(f"🧪 Testing fused model at {model_path}")
    
    try:
        model, tokenizer = load(model_path)
        
        # Test with a simple chess position
        test_prompt = "FEN: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1\nMove:"
        
        response = generate(
            model, tokenizer,
            prompt=test_prompt,
            max_tokens=5
        )
        
        print(f"✅ Test successful!")
        print(f"Prompt: {test_prompt}")
        print(f"Response: {response}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    # Set your parameters here
    base_model = "meta-llama/Llama-3.2-3B"  # Replace with your base model
    adapter_path = "chess_adapters"
    output_path = "chess_bot_fused"
    
    # Fuse the model
    #fuse_chess_model(base_model, adapter_path, output_path)
    
    # Test it
    test_fused_model(output_path)