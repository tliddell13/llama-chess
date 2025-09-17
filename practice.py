from mlx_lm import load, generate
import mlx.core as mx

# Load with memory optimization
model, tokenizer = load("meta-llama/Llama-3.2-3B")

# Check memory usage after loading
print(f"Memory used: {mx.metal.get_cache_memory() / 1024**3:.2f} GB")

response = generate(
    model, 
    tokenizer, 
    prompt="1. Nf3 Nf6 2. c4 g6 3. Nc3 d5 4. cxd5 Nxd5 5. g3 Bg7 6. Nxd5 Qxd5 7. Bg2 O-O 8. O-O Nc6 9. d3 Qd8 10. a3 e5 11. Bg5 Qd6 12. Qc2 Bg4 13. Be3 Rfe8 14. Rac1 Rac8 15. Rfe1 Ne7 16. Ng5 Nd5 17. Qb3 c6 18. Bxa7 Qe7 19. h4 h6 20. Bc5 Qd7 21. Ne4 b6 22. Bb4 Be6 23. Qa4 Red8 24. Bd2 f5 25. Nc3 Ne7 26. Red1 Kh7 27.",
    max_tokens=100
)
print(response)