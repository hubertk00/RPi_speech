import tensorflow as tf
TF_ENABLE_ONEDNN_OPTS=0
import librosa
import numpy as np
import matplotlib.pyplot as plt
from src.specaugment import SpecAugment
import random

def mfcc(audio, sr, n_mfcc=20, frame_length=400, frame_step=160):
    if not isinstance(audio, tf.Tensor):
        audio = tf.convert_to_tensor(audio, dtype=tf.float32)
    
    stfts = tf.signal.stft(
            audio,
            frame_length=frame_length,
            frame_step=frame_step,
            fft_length=512,
            window_fn=tf.signal.hann_window
    )
    
    spectrograms = tf.abs(stfts) #spektrogram amplitudowy
    num_spectrogram_bins = stfts.shape[-1] #ostatni wymiar tensora [batch_size, time_steps, num_frequencies-liczba binow czestotliwosciowych]
    lower_edge, upper_edge, num_mel_bins = 80.0, sr / 2, 40 #40 filtrow melowych
    
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix( #macierz wagowa liniowa -> skala melowa
        num_mel_bins, num_spectrogram_bins, sr, lower_edge, upper_edge)
    mel_spectrograms = tf.tensordot( #mnozenie macierzy
        spectrograms, linear_to_mel_weight_matrix, 1)
    log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6) / tf.math.log(tf.constant(10, dtype=tf.float32)) #log z mel-spektrogramu
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms) #obliczanie mfcc ze spektrogramu log
    mfccs = mfccs[..., :n_mfcc]

    return mfccs, log_mel_spectrograms

def porownanie_sygnalow(signal, rozszerzony, sr):
    czas_signal = np.arange(len(signal)) / sr
    czas_rozszerzony = np.arange(len(rozszerzony)) / sr
    fig, ax = plt.subplots(nrows=2)
    ax[0].plot(czas_signal, signal)
    ax[0].set(title="Oryginal", xlabel="Czas [s]", ylabel="Amplituda")
    ax[1].plot(czas_rozszerzony, rozszerzony)
    ax[1].set(title="Rozszerzony", xlabel="Czas [s]", ylabel="Amplituda")
    plt.tight_layout()
    plt.show()

def trim_pad(signal, sr, length):
    length_samples = int(length*sr)
    signal_length = len(signal)
    if signal_length > length_samples:
        signal = signal[:length_samples]
    elif signal_length < length_samples:
        padding_length = length_samples - signal_length
        padding = np.zeros(padding_length)
        signal = np.concatenate((signal, padding))
    return signal
    

def load_audio(signal, sr, duration):
    signal, _ = librosa.load(signal, sr=sr)
    signal = librosa.util.normalize(signal)
    signal = trim_pad(signal, sr, duration)

    return tf.convert_to_tensor(signal, dtype=tf.float32)

def mfcc_librosa(audio, sr=16000, n_mfcc=20):
    if not isinstance(audio, np.ndarray):
        audio = np.array(audio)
    
    n_fft = 512       
    hop_length = 160   
    win_length = 400   
    
    mfccs = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window='hann'
    )    
    mfccs = mfccs.T
    
    return mfccs

def apply_spec_augment(mfccs, label):
    spec_augmenter = SpecAugment()
    if isinstance(mfccs, np.ndarray):
        mfccs = tf.convert_to_tensor(mfccs, dtype=tf.float32)

    freq_mask_params = {'max_mask_width': 6, 'num_masks': 2}
    time_mask_params = {'max_mask_width': 8, 'num_masks': 2}

    augmentation_type = random.random()
    spec_augment_config = []

    if augmentation_type < 0.33:
        spec_augment_config.append({'name': 'frequency_mask', 'p': 1.0, 'params': freq_mask_params})
    elif augmentation_type < 0.66:
        spec_augment_config.append({'name': 'time_mask', 'p': 1.0, 'params': time_mask_params})
    else:
        spec_augment_config.append({'name': 'frequency_mask', 'p': 1.0, 'params': freq_mask_params})
        spec_augment_config.append({'name': 'time_mask', 'p': 1.0, 'params': time_mask_params})

    augmented_mfccs = spec_augmenter.rozszerz(mfccs, spec_augment_config)
    return augmented_mfccs, label

def plot_history(history):
    plt.figure(figsize=(10, 6))
    
    ax1 = plt.gca()
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy', color='blue')
    
    train_acc_line, = ax1.plot(history.history['accuracy'], 'b-', label='Train Accuracy')
    val_acc_line, = ax1.plot(history.history['val_accuracy'], 'b--', label='Val Accuracy')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Loss', color='red')
    
    train_loss_line, = ax2.plot(history.history['loss'], 'r-', label='Train Loss')
    val_loss_line, = ax2.plot(history.history['val_loss'], 'r--', label='Val Loss')
    ax2.tick_params(axis='y', labelcolor='red')
    
    lines = [train_acc_line, val_acc_line, train_loss_line, val_loss_line]
    labels = [line.get_label() for line in lines]
    plt.legend(lines, labels, loc='center right')
    
    plt.title('Training and Validation Metrics')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def extract_mfcc(audio, sr, n_mfcc):
    audio = librosa.util.normalize(audio)
    mfccs, _ = mfcc(audio, sr=sr, n_mfcc=n_mfcc, frame_length=400, frame_step=160)
    return mfccs

if __name__ == "__main__":
    signal = load_audio(r"wake_word\Hugo\hugo_20.wav", 16000)
    mfccs, _ = mfcc(signal, 16000) #zwrocenie tylko mfccs
    print(mfccs.shape)
    