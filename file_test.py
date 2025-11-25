import argparse
import numpy as np
import tensorflow as tf
import librosa
from keras.api.models import load_model
from src.funkcje import mfcc, load_audio

COMMANDS = ['Ciemniej', 'Jasniej', 'Muzyka', 'Rolety', 'Swiatlo', 'Telewizor', 'Wrocilem', 'Wychodze', 'Tlo']

def extract_mfcc(audio, sr, n_mfcc):
    audio = librosa.util.normalize(audio)
    mfccs, _ = mfcc(audio, sr=sr, n_mfcc=n_mfcc, frame_length=400, frame_step=160)
    return mfccs

def predict_audio(audio_path, model, is_command, sr=16000, n_mfcc=20, command_threshold=0.9, wake_threshold=0.7):
    audio = load_audio(audio_path, sr=sr, duration=1.0)
    mfcc_features = extract_mfcc(audio, sr, n_mfcc)
    input_tensor = tf.expand_dims(mfcc_features, axis=0)

    predictions = model.predict(input_tensor, verbose=0)

    if is_command:
        max_prob = np.max(predictions[0])
        index = np.argmax(predictions[0])
        label = COMMANDS[index]
        print(f"Wykryto: {label} (Prawdopodobieństwo: {max_prob:.2f})" if max_prob > command_threshold else f"Brak detekcji (max={max_prob:.2f})")
    else:
        prob = predictions[0][0]
        print(f"Wykryto Hugo (Prawdopodobieństwo: {prob:.2f})" if prob > wake_threshold else f"Brak detekcji (p={prob:.2f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Testowanie rozpoznawania mowy z pliku .wav")

    parser.add_argument("--audio_path", type=str, help="Ścieżka do pliku .wav")
    parser.add_argument("--model_path", type=str, help="Ścieżka do modelu (.keras)")
    parser.add_argument("--type", choices=["wake", "command"], required=True, help="Typ modelu: wake lub command")
    parser.add_argument("--rate", type=int, default=16000, help="Częstotliwość próbkowania audio")
    parser.add_argument("--n-mfcc", type=int, default=20, help="Liczba MFCC")
    parser.add_argument("--wake-threshold", type=float, default=0.7)
    parser.add_argument("--command-threshold", type=float, default=0.9)

    args = parser.parse_args()

    model = load_model(args.model_path)

    predict_audio(
        audio_path=args.audio_path,
        model=model,
        is_command=(args.type == "command"),
        sr=args.rate,
        n_mfcc=args.n_mfcc,
        command_threshold=args.command_threshold,
        wake_threshold=args.wake_threshold
    )
