import tensorflow as tf
import tf2onnx
import numpy as np

KERAS_MODEL_PATH = r'C:\Users\Hubert\Desktop\Praca_dyplomowa_TensorFlow\trained_models_commands\resnet14_commands.keras'
ONNX_OUTPUT_PATH = 'resnet14.onnx'

model = tf.keras.models.load_model(KERAS_MODEL_PATH)

input_signature = [
    tf.TensorSpec((None, None, 20, 1), tf.float32, name='input_audio')
    ]

# 4. Konwersja
print(f"Konwertuję model {KERAS_MODEL_PATH} do ONNX...")
model_proto, _ = tf2onnx.convert.from_keras(
    model, 
    input_signature=input_signature, 
    opset=13, 
    output_path=ONNX_OUTPUT_PATH
)

print(f"Gotowe! Zapisano jako: {ONNX_OUTPUT_PATH}")