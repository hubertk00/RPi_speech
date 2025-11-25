import sounddevice as sd
import numpy as np
import tensorflow as tf
import queue
import threading
import time
import keras.api.models
import librosa
import argparse
from src.funkcje import mfcc, extract_mfcc

TF_ENABLE_ONEDNN_OPTS = 0

COMMANDS = ['Ciemniej', 'Jasniej', 'Muzyka', 'Rolety', 'Swiatlo', 'Telewizor', 'Wrocilem', 'Wychodze', 'Tlo']

q = queue.Queue()
running = True


def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"Audio callback status: {status}")
    q.put(indata.copy().flatten())

def process_audio(wake_model, command_model, args):
    global running
    
    audio_buffer = np.zeros(args.window_size, dtype=np.float32)
    last_detection_time = 0
    listening_for_command = False
    command_mode_start_time = 0

    while running:
        try:
            data = q.get(timeout=1)
            audio_buffer = np.roll(audio_buffer, -len(data))
            audio_buffer[-len(data):] = data

            current_time = time.time()

            if listening_for_command:
                if current_time - command_mode_start_time > args.listen_duration:
                    listening_for_command = False
                    print("\nZakonczono nasluchiwanie komendy.")
                    continue

                mfcc_features = extract_mfcc(audio_buffer, args.rate, args.n_mfcc)
                input_tensor = tf.expand_dims(mfcc_features, axis=0)
                predictions = command_model.predict(input_tensor, verbose=0)[0]
                
                max_prob = np.max(predictions)
                command_index = np.argmax(predictions)

                if max_prob > args.command_threshold:
                    command_name = COMMANDS[command_index]
                    if command_name != 'Tlo':
                        print(f"\nWykryto: {command_name} (Prawdopodobienstwo: {max_prob:.2f})")
                        listening_for_command = False 
                    else:
                        print(".", end="", flush=True)
                else:
                    print(".", end="", flush=True)

            else: 
                if current_time - last_detection_time < args.cooldown:
                    continue
                
                mfcc_features = extract_mfcc(audio_buffer, args.rate, args.n_mfcc)
                input_tensor = tf.expand_dims(mfcc_features, axis=0)
                prediction = wake_model.predict(input_tensor, verbose=0)[0][0]

                if prediction > args.wake_threshold:
                    print(f"\nWykryto Hugo (Prawdopodobienstwo: {prediction:.2f})")
                    print("Nasluchuje komendy...")
                    listening_for_command = True
                    command_mode_start_time = current_time
                    last_detection_time = current_time
                else:
                    print(".", end="", flush=True)

        except queue.Empty:
            pass

def main(args):
    global running

    try:
        wake_model = keras.api.models.load_model(args.wake_model)
        command_model = keras.api.models.load_model(args.command_model)

        processing_thread = threading.Thread(target=process_audio, args=(wake_model, command_model, args))
        processing_thread.start()

        print("\nNasluchiwanie slowa wybudzajacego...")
        with sd.InputStream(callback=audio_callback,
                            channels=1,
                            samplerate=args.rate,
                            blocksize=args.chunk,
                            dtype='float32'):
            while running:
                time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nPrzerwano dzialanie systemu.")
    except Exception as e:
        print(f"\n[BŁĄD]: {e}")
    finally:
        running = False
        if 'processing_thread' in locals() and processing_thread.is_alive():
            processing_thread.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="System rozpoznawania mowy")
    
    parser.add_argument('-wm', '--wake-model', type=str, required=True, help="Ścieżka do zapisanego modelu słowa wybudzającego (.keras)")
    parser.add_argument('-cm', '--command-model', type=str, required=True, help="Ścieżka do zapisanego modelu komend (.keras)")
    
    parser.add_argument('-wt', '--wake-threshold', type=float, default=0.8, help="Próg pewności dla detekcji słowa wybudzającego.")
    parser.add_argument('-ct', '--command-threshold', type=float, default=0.9, help="Próg pewności dla detekcji komendy.")
    parser.add_argument('--cooldown', type=float, default=1.0, help="Czas oczekiwania po detekcji.")
    parser.add_argument('--listen-duration', type=float, default=4.0, help="Czas nasłuchiwania komendy po wybudzeniu.")
    
    parser.add_argument('--rate', type=int, default=16000, help="Częstotliwość próbkowania audio.")
    parser.add_argument('--chunk', type=int, default=2048, help="Rozmiar ramki audio.")
    parser.add_argument('--window-size', type=int, default=16000, help="Rozmiar okna analizy (powinien odpowiadać 1 sekundzie audio).")
    parser.add_argument('--n-mfcc', type=int, default=20, help="Liczba współczynników MFCC do ekstrakcji.")

    args = parser.parse_args()
    
    main(args)