import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import keras.api.models

resnet8_wake_word = keras.api.models.load_model(r'C:\Users\Hubert\Desktop\Praca_dyplomowa_TensorFlow\trained_models_wake_word\resnet8_wake_word.keras')
resnet14_wake_word = keras.api.models.load_model(r'C:\Users\Hubert\Desktop\Praca_dyplomowa_TensorFlow\trained_models_wake_word\resnet14_wake_word.keras')
matchboxnet_wake_word = keras.api.models.load_model(r'C:\Users\Hubert\Desktop\Praca_dyplomowa_TensorFlow\trained_models_wake_word\matchboxnet_wake_word.keras')
crnn_wake_word = keras.api.models.load_model(r'C:\Users\Hubert\Desktop\Praca_dyplomowa_TensorFlow\trained_models_wake_word\crnn_wake_word.keras')

resnet8_commands = keras.api.models.load_model(r'C:\Users\Hubert\Desktop\Praca_dyplomowa_TensorFlow\trained_models_commands\resnet8_commands.keras')
resnet14_commands = keras.api.models.load_model(r'C:\Users\Hubert\Desktop\Praca_dyplomowa_TensorFlow\trained_models_commands\resnet14_commands.keras')
crnn_commands = keras.api.models.load_model(r'C:\Users\Hubert\Desktop\Praca_dyplomowa_TensorFlow\trained_models_commands\crnn_commands.keras')
matchboxnet_commands = keras.api.models.load_model(r'C:\Users\Hubert\Desktop\Praca_dyplomowa_TensorFlow\trained_models_commands\matchboxnet_commands.keras')

converter = tf.lite.TFLiteConverter.from_keras_model(crnn_commands)
tflite_model = converter.convert()
with open('crnn_commands.tflite', 'wb') as f:
    f.write(tflite_model)