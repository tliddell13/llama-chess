# Convert the PGN files to FEN format
import chess
import chess.pgn
import json
from pathlib import Path
from typing import List, Dict

# Turn every game into several training positions
def extract_positions_from_game(game) -> List[Dict]:
    positions = []
    board = game.board()

    for move in game.mainline_moves():

        # Get the current position in FEN format
        fen = board.fen()

        # Get the move in SAN notation
        move_san = board.san(move)

        # Create training example: FEN -> move
        positions.append({
            "text": f"{fen} -> {move_san}"
        })
        
        # Make the move
        board.push(move)

    return positions

def process_pgn_file(pgn_path: Path) -> List[Dict]:
    #Process a single PGN file into training examples
    positions = []
    games_processed = 0
    
    print(f"Processing {pgn_path.name}...")
    
    with open(pgn_path, 'r', encoding='utf-8', errors='ignore') as pgn_file:
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
            
            games_processed += 1
            
            # Extract positions from this game
            game_positions = extract_positions_from_game(game)
            positions.extend(game_positions)
    
    print(f"  Games: {games_processed}")
    print(f"  Training examples: {len(positions)}")
    
    return positions

def process_directory(input_dir: Path, output_file: Path):
    #Process all PGN files in a directory
    all_positions = []
    
    # Find all PGN files
    pgn_files = list(input_dir.glob("*.pgn"))
    
    print(f"Found {len(pgn_files)} PGN files\n")
    
    for pgn_path in pgn_files:
        positions = process_pgn_file(pgn_path)
        all_positions.extend(positions)
    
    # Save as JSONL
    print(f"\nSaving {len(all_positions)} training examples to {output_file}...")
    with open(output_file, 'w') as f:
        for position in all_positions:
            f.write(json.dumps(position) + '\n')
    
    print(f"Done!")
    print(f"\nSummary:")
    print(f"  Total positions: {len(all_positions):,}")
    
    # Show a few examples
    print(f"\nExample training data:")
    for i in range(min(3, len(all_positions))):
        print(f"  {all_positions[i]['text']}")

if __name__ == "__main__":
    input_directory = Path("pgn_files")
    output_file = Path("chess_training.jsonl")
    
    process_directory(input_directory, output_file)
    