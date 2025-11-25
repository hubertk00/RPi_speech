import matplotlib.pyplot as plt
import librosa
import tensorflow as tf
import random
from src.funkcje import mfcc
from src.specaugment import SpecAugment

rozszerzenie = SpecAugment()

def visualize_masks(audio_path, max_freq_mask_width=3, freq_masks=2, max_time_mask_width=3, time_masks=2):
    audio, sr = librosa.load(audio_path, sr=None)
    
    features, mel_spectrogram = mfcc(audio, sr, n_mfcc=13)
    
    features_np = features.numpy()
    mel_spec_np = mel_spectrogram.numpy()
    
    freq_masked_features = rozszerzenie.frequency_mask(features, max_freq_mask_width, freq_masks)
    freq_masked_mel = rozszerzenie.frequency_mask(mel_spectrogram, max_freq_mask_width, freq_masks)
    
    time_masked_features = rozszerzenie.time_mask(features, max_time_mask_width, time_masks)
    time_masked_mel = rozszerzenie.time_mask(mel_spectrogram, max_time_mask_width, time_masks)
    
    both_masked_features = rozszerzenie.time_mask(rozszerzenie.frequency_mask(features, max_freq_mask_width, freq_masks), max_time_mask_width, time_masks)
    both_masked_mel = rozszerzenie.time_mask(rozszerzenie.frequency_mask(mel_spectrogram, max_freq_mask_width, freq_masks), max_time_mask_width, time_masks)
    
    freq_masked_features_np = freq_masked_features.numpy()
    freq_masked_mel_np = freq_masked_mel.numpy()
    time_masked_features_np = time_masked_features.numpy()
    time_masked_mel_np = time_masked_mel.numpy()
    both_masked_features_np = both_masked_features.numpy()
    both_masked_mel_np = both_masked_mel.numpy()
    
    plt.figure(figsize=(16, 20))
    
    plt.subplot(4, 2, 1)
    librosa.display.specshow(mel_spec_np.T, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Oryginalny mel-spektrogram')
    
    plt.subplot(4, 2, 2)
    librosa.display.specshow(freq_masked_mel_np.T, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-spektrogram z maskami częstotliwościowymi')
    
    plt.subplot(4, 2, 3)
    librosa.display.specshow(time_masked_mel_np.T, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-spektrogram z maskami czasowymi')
    
    plt.subplot(4, 2, 4)
    librosa.display.specshow(both_masked_mel_np.T, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-spektrogram z oboma rodzajami masek')
    
    plt.subplot(4, 2, 5)
    librosa.display.specshow(features_np.T, sr=sr, x_axis='time')
    plt.colorbar()
    plt.title('Oryginalne MFCC')
    
    plt.subplot(4, 2, 6)
    librosa.display.specshow(freq_masked_features_np.T, sr=sr, x_axis='time')
    plt.colorbar()
    plt.title('MFCC z maskami częstotliwościowymi')
    
    plt.subplot(4, 2, 7)
    librosa.display.specshow(time_masked_features_np.T, sr=sr, x_axis='time')
    plt.colorbar()
    plt.title('MFCC z maskami czasowymi')
    
    plt.subplot(4, 2, 8)
    librosa.display.specshow(both_masked_features_np.T, sr=sr, x_axis='time')
    plt.colorbar()
    plt.title('MFCC z oboma rodzajami masek')
    
    plt.tight_layout()
    plt.savefig('masked_spectrograms.png', dpi=300)
    plt.show()
    
    return {
        'original_mel': mel_spec_np,
        'freq_masked_mel': freq_masked_mel_np,
        'time_masked_mel': time_masked_mel_np,
        'both_masked_mel': both_masked_mel_np,
        'original_mfcc': features_np,
        'freq_masked_mfcc': freq_masked_features_np,
        'time_masked_mfcc': time_masked_features_np,
        'both_masked_mfcc': both_masked_features_np
    }
    
visualize_masks("hugo_008.wav", max_freq_mask_width=3, freq_masks=4, max_time_mask_width=3, time_masks=4)