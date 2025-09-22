import requests
from pathlib import Path
import time

def download_pgn_files(output_dir="pgn_files", delay=1):
    """
    Download specific PGN files from pgnmentor.com
    
    The files are at: pgnmentor.com/players/[PlayerName].zip

    You can browse the website and pick other players if you want
    """
    
    # Elite players for Phase 1
    players = {
        "Carlsen": "Magnus Carlsen",
        "Nakamura": "Hikaru Nakamura",
        "Caruana": "Fabiano Caruana"
    }
    
    base_url = "https://www.pgnmentor.com/players/"
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"Downloading PGN files for elite players...\n")
    
    for filename, full_name in players.items():
        url = f"{base_url}{filename}.zip"
        output_file = output_path / f"{filename}.zip"
        
        # Skip if already downloaded
        if output_file.exists():
            print(f"✓ {full_name}: Already downloaded")
            continue
        
        try:
            print(f"Downloading {full_name}...", end=" ")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(output_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            file_size = output_file.stat().st_size / (1024 * 1024)  # MB
            print(f"✓ ({file_size:.1f} MB)")
            time.sleep(delay)  # Be polite to the server
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                print(f"✗ File not found at {url}")
            else:
                print(f"✗ Error: {e}")
        except Exception as e:
            print(f"✗ Error: {e}")
    
    print(f"\nDownload complete! Files saved to {output_path.absolute()}")
    
    # List what we got
    downloaded = list(output_path.glob("*.zip"))
    print(f"\nDownloaded files:")
    for f in downloaded:
        size = f.stat().st_size / (1024 * 1024)
        print(f"  - {f.name} ({size:.1f} MB)")

if __name__ == "__main__":
    download_pgn_files(output_dir="pgn_files", delay=1)

