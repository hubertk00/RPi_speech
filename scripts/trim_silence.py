import librosa
import numpy
import os
from pathlib import Path
from src.funkcje import trim_pad
import soundfile as sf

def trim_silence(audio, sr=16000, top_db=20, frame_length=400, hop_length=160):
    trimmed_audio, _ = librosa.effects.trim(
        audio,
        top_db=top_db,
        frame_length=frame_length,
        hop_length=hop_length)
    return trimmed_audio

def trim_dataset(commands, input_path, output_path, sr=16000, target_length=2.0):
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True)

    for command in commands:
        input_dir = input_path / command
        output_dir = output_path / command
        output_dir.mkdir(exist_ok=True)

        for file_name in os.listdir(input_dir):
            if not file_name.endswith('.wav'):
                continue

            file_path = input_dir / file_name
            output_file_path = output_dir / file_name

            audio, _ = librosa.load(file_path, sr=sr)
            trimmed_audio = trim_silence(audio, sr=sr)

            trimmed_audio = trim_pad(trimmed_audio, sr, target_length)

            sf.write(output_file_path, trimmed_audio, sr)

if __name__ == "__main__":
    commands = ['Swiatlo', 'Jasniej', 'Ciemniej', 'Wychodze', 'Wrocilem', 'Rolety_gora', 'Rolety_dol', 'Telewizor', 'Wlacz_muzyke', 'Wylacz_muzyke']
    input_path = r'C:\Users\Hubert\Desktop\Praca_jupyter\Nagrania'
    output_path = r'C:\Users\Hubert\Desktop\Praca_jupyter\Nagrania_trim'
    trim_dataset(commands, input_path, output_path, sr=16000, target_length=2.0)



