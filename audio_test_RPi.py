import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import argparse
import threading
import queue
import time
import numpy as np
import sounddevice as sd
import psutil
import csv
from src.funkcje import extract_mfcc

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        import tensorflow.lite as tflite
    except ImportError:
        exit(1)

COMMANDS = ['Ciemniej', 'Jasniej', 'Muzyka', 'Rolety', 'Swiatlo', 'Telewizor', 'Wrocilem', 'Wychodze', 'Tlo']

q = queue.Queue()
running = True

class ResourceMonitor:
    def __init__(self, output_file='system_usage_TF.csv', interval=0.5):
        self.output_file = output_file
        self.interval = interval
        self.running = False
        self.process = psutil.Process(os.getpid())
        self.thread = threading.Thread(target=self._monitor_loop)

    def start(self):
        self.running = True
        with open(self.output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Time_s', 'CPU_Percent', 'RAM_MB'])
        self.thread.start()
    
    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()
    
    def _monitor_loop(self):
        start_time = time.time()
        while self.running:
            try:
                cpu = self.process.cpu_percent(interval=None)
                mem_info = self.process.memory_info()
                ram_mb = mem_info.rss / (1024 * 1024) 
                
                elapsed = time.time() - start_time
                
                with open(self.output_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([f"{elapsed:.2f}", cpu, f"{ram_mb:.2f}"])
                
                time.sleep(self.interval)
            except Exception as e:
                print(f"Błąd monitora: {e}")
                break

class TFLiteModel:
    def __init__(self, model_path):
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_index = self.input_details[0]['index']
        self.output_index = self.output_details[0]['index']

    def predict(self, input_data):
        if input_data.ndim == 2:
            input_data = np.expand_dims(input_data, axis=0)
        
        input_data = input_data.astype(np.float32)
        
        self.interpreter.set_tensor(self.input_index, input_data)
        self.interpreter.invoke()
        
        return self.interpreter.get_tensor(self.output_index)

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

    wake_times = []
    cmd_times = []

    while running:
        try:
            data = q.get(timeout=1)
            audio_buffer = np.roll(audio_buffer, -len(data))
            audio_buffer[-len(data):] = data

            current_time = time.time()
            
            mfcc_features = extract_mfcc(audio_buffer, args.rate, args.n_mfcc)
            
            if listening_for_command:
                if current_time - command_mode_start_time > args.listen_duration:
                    listening_for_command = False
                    print("\nZakonczono nasluchiwanie komendy.")
                    continue

                t0 = time.perf_counter()
                
                predictions = command_model.predict(mfcc_features)[0]
                
                t1 = time.perf_counter()
                
                cmd_times.append(t1 - t0)  
                if len(cmd_times) > 0 and len(cmd_times) % 100 == 0:
                    avg_cmd = sum(cmd_times[-100:]) / 100
                    print(f"Model Komend: {avg_cmd*1000:.2f} ms")     

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
               
                t0_wake = time.perf_counter()
                
                prediction_out = wake_model.predict(mfcc_features)[0]
                
                t1_wake = time.perf_counter()

                wake_times.append(t1_wake - t0_wake)
                if len(wake_times) > 0 and len(wake_times) % 100 == 0:
                    avg_wake = sum(wake_times[-100:]) / 100
                    print(f"Model Wake: {avg_wake*1000:.2f} ms")
                                
                prediction = prediction_out[0] if np.ndim(prediction_out) > 0 else prediction_out

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
        except Exception as e:
            print(f"Błąd w pętli: {e}")

def main(args):
    global running
    monitor = ResourceMonitor(output_file='system_usage.csv', interval=0.5)
    monitor.start()
    try:
        wake_model = TFLiteModel(args.wake_model)
        command_model = TFLiteModel(args.command_model)

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
        monitor.stop()
        if 'processing_thread' in locals() and processing_thread.is_alive():
            processing_thread.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-wm', '--wake-model', type=str, required=True, help="Ścieżka do modelu .tflite")
    parser.add_argument('-cm', '--command-model', type=str, required=True, help="Ścieżka do modelu .tflite")
    
    parser.add_argument('-wt', '--wake-threshold', type=float, default=0.8, help="Próg dla detekcji słowa wybudzającego.")
    parser.add_argument('-ct', '--command-threshold', type=float, default=0.9, help="Próg dla detekcji komendy.")
    parser.add_argument('--cooldown', type=float, default=1.0, help="Czas oczekiwania po detekcji.")
    parser.add_argument('--listen-duration', type=float, default=4.0, help="Czas nasłuchiwania komendy po wybudzeniu.")
    
    parser.add_argument('--rate', type=int, default=16000, help="Częstotliwość próbkowania audio.")
    parser.add_argument('--chunk', type=int, default=2048, help="Rozmiar ramki audio.")
    parser.add_argument('--window-size', type=int, default=16000, help="Rozmiar okna.")
    parser.add_argument('--n-mfcc', type=int, default=20, help="Liczba współczynników MFCC do ekstrakcji.")

    parser.add_argument('--vad-threshold', type=float, default=0.005)

    args = parser.parse_args()
    
    main(args)