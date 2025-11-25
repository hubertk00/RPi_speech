import os
TF_ENABLE_ONEDNN_OPTS=0
import numpy as np
import librosa
from IPython.display import Audio
import random
import soundfile as sf
import tensorflow as tf

class Rozszerzanie:
    def __init__(self):
        self.transforms = {
            'add_white_noise': self.add_white_noise,
            'time_stretch': self.time_stretch,
            'pitch_scale': self.pitch_scale,
            'random_gain': self.random_gain,
            'add_noise': self.add_noise
        }

        self.noise_library = {} #przechowywanie dzwiekow tla
    def add_noise_to_library(self, name, noise_signal):
        self.noise_library[name] = noise_signal
        

    def apply_augmentation(self, audio, label, configs, debug=False):
        if isinstance(audio, tf.Tensor):
            audio = audio.numpy()

        sr = 16000
        rozszerzone_audio = self.rozszerz(audio, sr, configs)
        if debug:
            os.makedirs("debug_audio", exist_ok=True)
            file_name = f"debug_audio/augmented_test{random.randint(0,9999)}.wav"
            sf.write(file_name, rozszerzone_audio, sr)

        return tf.convert_to_tensor(rozszerzone_audio, dtype=tf.float32), label
    
    def normalize_audio(self, signal):
        return signal / np.max(np.abs(signal)) if np.max(np.abs(signal)) > 0 else signal

    def rozszerz(self, signal, sr, configs):
        rozszerzony = np.copy(signal)
        for config in configs:
            name = config['name']
            p = config.get('p', 1.0)
            params = config.get('params', {})
            if name == 'add_noise':
                if 'noise_options' in params:
                    noise_options = params.pop('noise_options', None) #pop pomaga usunac z parametrow noise options i przekazac je do zmiennej
                    if random.random() < p:
                        noise_name = [opt[0] for opt in noise_options]
                        weights = [opt[1] for opt in noise_options]

                        total_weight = sum(weights)
                        if total_weight > 0:
                            weights = [w/total_weight for w in weights]
                        selected_noise = random.choices(noise_name, weights = weights, k=1)[0]
                        if selected_noise in self.noise_library:
                            selected_noise = self.noise_library[selected_noise]
                            rozszerzony = self.add_noise(rozszerzony, selected_noise, **params)
            elif name in self.transforms and random.random() < p:
                transform_func = self.transforms[name]
                rozszerzony = transform_func(rozszerzony, sr, **params)
        return rozszerzony
        
    def add_white_noise(self, signal, sr, noise_factor):
        noise = np.random.normal(0, signal.std(), signal.size)
        zaszumiony = signal + noise * noise_factor
        return zaszumiony

    def time_stretch(self, signal, sr, stretch_rate, target_duration):
        stretched_signal = librosa.effects.time_stretch(signal, rate=stretch_rate)
        target_len = int(sr*target_duration)
        if len(stretched_signal) > target_len:
            stretched_signal = stretched_signal[:target_len]
        elif len(stretched_signal) < target_len:
            padding_length = target_len - len(stretched_signal)
            stretched_signal = np.pad(stretched_signal, (0, padding_length), mode = 'constant')
        return stretched_signal

    def pitch_scale(self, signal, sr, num_semitones): 
        #na + skala idzie w gore, na - skala idzie w dol
        return librosa.effects.pitch_shift(signal, sr=sr, n_steps=num_semitones)

    def random_gain(self, signal, sr, min_gain, max_gain):
        gain_factor = random.uniform(min_gain, max_gain)
        return signal * gain_factor

    def add_noise(self, signal, noise, snr):
        if len(noise) < len(signal):
            noise_repeated = np.tile(noise, int(np.ceil(len(signal) / len(noise)))) #powtarzanie noise, aby dopelnil czas sygnalu
            noise = noise_repeated[:len(signal)]
        else: #jesli szum dluzszy to losowy segment
            start = np.random.randint(0, len(noise) - len(signal) + 1) #dodane plus 1 bo indeks high jest wykluczony w randint
            noise = noise[start:start + len(signal)] #start-losowy fragment gdzie zaczyna, start+len koniec

        audio_power = np.sum(signal**2)/len(signal)
        noise_power = np.sum(noise**2)/len(noise)

        if noise_power==0:
            return signal

        noise_adjusted = noise * np.sqrt(audio_power / (10**(snr/10) * noise_power))
        audio_noisy = signal + noise_adjusted
        if np.max(np.abs(audio_noisy)) > 1.0: 
            audio_noisy = audio_noisy / np.max(np.abs(audio_noisy))

        return audio_noisy
