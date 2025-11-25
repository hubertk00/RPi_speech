import os
TF_ENABLE_ONEDNN_OPTS = 0
import numpy as np
import tensorflow as tf
import random

class SpecAugment:
    def __init__(self):
        self.mfcc_transforms = {
            'frequency_mask': self.frequency_mask,
            'time_mask': self.time_mask
        }

    def rozszerz(self, features, configs):
        rozszerzony = tf.identity(features)
        for config in configs:
            name = config['name']
            p = config.get('p', 1.0)
            params = config.get('params', {})
            if name in self.mfcc_transforms and random.random() < p:
                transform_func = self.mfcc_transforms[name]
                rozszerzony = transform_func(rozszerzony, **params)
        return rozszerzony

    def frequency_mask(self, features, max_mask_width, num_masks):
        masked = tf.identity(features)

        if len(features.shape) == 2:  # [time, freq]
            time_dim, freq_dim = tf.shape(features)[0], tf.shape(features)[1]
            for i in range(num_masks):
                f = tf.random.uniform(shape=[], minval=1, maxval=max_mask_width, dtype=tf.int32)  # losowa szerokość maski
                f0 = tf.random.uniform(shape=[], minval=0, maxval=freq_dim - f, dtype=tf.int32)  # częstotliwość początkowa

                mask = tf.concat([  # połączenie wzdłuż osi kolumn
                    tf.ones([time_dim, f0], tf.float32),  # przed maską wypełniona 1
                    tf.zeros([time_dim, f], tf.float32),  # maska wypełniona 0
                    tf.ones([time_dim, freq_dim - f0 - f], tf.float32)  # część za maską wypełniona 1
                ], axis=1)  # axis 1 to częstotliwość, jeśli mamy 2 wymiary
                masked = masked * mask

        else:  # [batch, time, freq]
            batch_size, time_dim, freq_dim = tf.shape(features)[0], tf.shape(features)[1], tf.shape(features)[2]
            for i in range(num_masks):
                f = tf.random.uniform(shape=[], minval=1, maxval=max_mask_width, dtype=tf.int32)
                f0 = tf.random.uniform(shape=[], minval=0, maxval=freq_dim - f, dtype=tf.int32)
                mask = tf.concat([
                    tf.ones([batch_size, time_dim, f0], tf.float32),
                    tf.zeros([batch_size, time_dim, f], tf.float32),
                    tf.ones([batch_size, time_dim, freq_dim - f0 - f], tf.float32)
                ], axis=2)
                masked = masked * mask
        return masked

    def time_mask(self, features, max_mask_width, num_masks):
        masked = tf.identity(features)

        if len(features.shape) == 2:  # [time, freq]
            time_dim, freq_dim = tf.shape(features)[0], tf.shape(features)[1]
            for i in range(num_masks):
                t = tf.random.uniform(shape=[], minval=1, maxval=max_mask_width, dtype=tf.int32)  # losowa szerokość maski
                t0 = tf.random.uniform(shape=[], minval=0, maxval=time_dim - t, dtype=tf.int32)  # czas początkowy

                mask = tf.concat([
                    tf.ones([t0, freq_dim], tf.float32),
                    tf.zeros([t, freq_dim], tf.float32),
                    tf.ones([time_dim - t0 - t, freq_dim], tf.float32)
                ], axis=0)  # axis 0 to czas, jeśli mamy 2 wymiary
                masked = masked * mask
        else:  # [batch, time, freq]
            batch_size, time_dim, freq_dim = tf.shape(features)[0], tf.shape(features)[1], tf.shape(features)[2]
            for i in range(num_masks):
                t = tf.random.uniform(shape=[], minval=1, maxval=max_mask_width, dtype=tf.int32)
                t0 = tf.random.uniform(shape=[], minval=0, maxval=time_dim - t, dtype=tf.int32)
                mask = tf.concat([
                    tf.ones([batch_size, t0, freq_dim], tf.float32),
                    tf.zeros([batch_size, t, freq_dim], tf.float32),
                    tf.ones([batch_size, time_dim - t0 - t, freq_dim], tf.float32)
                ], axis=1)
                masked = masked * mask
        return masked