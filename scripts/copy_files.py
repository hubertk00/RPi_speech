import os
import random
import shutil
import argparse

def copy_random_wav_files(source_dir, dest_dir, num_files=50):
    if not os.path.exists(source_dir):
        raise FileNotFoundError(f"Folder źródłowy {source_dir} nie istnieje")
    
    wav_files = [f for f in os.listdir(source_dir) if f.endswith('.wav')]
    
    if len(wav_files) < num_files:
        raise ValueError(f"W folderze źródłowym jest tylko {len(wav_files)} plików .wav, żądano {num_files}.")
    
    selected_files = random.sample(wav_files, num_files)
    
    os.makedirs(dest_dir, exist_ok=True)
    
    for file_name in selected_files:
        source_path = os.path.join(source_dir, file_name)
        dest_path = os.path.join(dest_dir, file_name)
        shutil.copy2(source_path, dest_path)
        print(f"Skopiowano {file_name} do {dest_dir}")

def main():
    parser = argparse.ArgumentParser(description='Kopiuje losowe pliki WAV z folderu źródłowego do docelowego')
    parser.add_argument('--source_dir', help='Ścieżka do folderu źródłowego z plikami WAV')
    parser.add_argument('--dest_dir', help='Ścieżka do folderu docelowego')
    parser.add_argument('--n', '--num_files', dest='num_files', type=int, default=50,
                       help='Liczba plików do skopiowania (domyślnie: 50)')
    
    args = parser.parse_args()
    
    try:
        copy_random_wav_files(args.source_dir, args.dest_dir, args.num_files)
        print(f"Pomyślnie skopiowano {args.num_files} plików WAV do {args.dest_dir}")
    except Exception as e:
        print(f"Błąd: {e}")
        exit(1)

if __name__ == "__main__":
    main()