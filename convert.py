import tensorflow as tf
import keras.api.models

resnet8_wake_word = keras.api.models.load_model(r'C:\Users\hubi2\Desktop\Praca_jupyter\RPi_speech\trained_models_wake_word\resnet8_wake_word.keras')
resnet14_wake_word = keras.api.models.load_model(r'C:\Users\hubi2\Desktop\Praca_jupyter\RPi_speech\trained_models_wake_word\resnet14_wake_word.keras')

resnet8_commands = keras.api.models.load_model(r'C:\Users\hubi2\Desktop\Praca_jupyter\RPi_speech\trained_models_commands\resnet8_commands.keras')
resnet14_commands = keras.api.models.load_model(r'C:\Users\hubi2\Desktop\Praca_jupyter\RPi_speech\trained_models_commands\resnet14_commands.keras')
crnn_commands = keras.api.models.load_model(r'C:\Users\hubi2\Desktop\Praca_jupyter\RPi_speech\trained_models_commands\crnn_commands.keras')

converter = tf.lite.TFLiteConverter.from_keras_model(resnet14_commands)
tflite_model = converter.convert()
with open('resnet14_commands.tflite', 'wb') as f:
    f.write(tflite_model)