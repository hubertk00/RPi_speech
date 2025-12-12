import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import time
import statistics
import argparse
import numpy as np
import sys
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        import tensorflow.lite as tflite
    except ImportError:
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help="Ścieżka do pliku .tflite")
    args = parser.parse_args()

    try:
        interpreter = tflite.Interpreter(model_path=args.model_path)
        interpreter.allocate_tensors()
    except Exception as e:
        print(f"Błąd ładowania modelu: {e}")
        sys.exit(1)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape']
    input_index = input_details[0]['index']
    output_index = output_details[0]['index']

    input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)

    for i in range(10):
        interpreter.set_tensor(input_index, input_data)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_index)

    num_iterations = 10000
    results = []

    for i in range(num_iterations):
        start_time = time.time()
        
        interpreter.set_tensor(input_index, input_data)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_index) 
        
        end_time = time.time()
        results.append(end_time - start_time)

    average = statistics.mean(results)
    print(f"Sredni czas inferencji: {average:.8f}")