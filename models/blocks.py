from keras.api import layers, regularizers

def residual_block_type_one(input_shape, c, k, l2_rate=0.001):
    multiplied_size = int(c * k)
    x = layers.Conv1D(multiplied_size, 3, strides=1, use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_rate))(input_shape)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv1D(multiplied_size, 3, strides=1, use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_rate))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, input_shape])
    x = layers.ReLU()(x)
    return x

def residual_block_type_two(input_shape, c, k, l2_rate=0.001):
    multiplied_size = int(c * k)
    x1 = layers.Conv1D(multiplied_size, 3, strides=2, use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_rate))(input_shape)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.ReLU()(x1)
    x1 = layers.Conv1D(multiplied_size, 3, strides=1, use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_rate))(x1)
    x1 = layers.BatchNormalization()(x1)

    x2 = layers.Conv1D(multiplied_size, 1, strides=2, use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_rate))(input_shape)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.ReLU()(x2)

    x = layers.Add()([x1, x2])
    x = layers.ReLU()(x)
    return x

def tcs_conv_step(x, out_channels, kernel_size):
    x = layers.SeparableConv1D(
        filters=out_channels,
        kernel_size=kernel_size,
        padding='same',
        use_bias=True
    )(x)
    return x

def matchbox_main_block(x, out_channels, kernel_size, R=1):
    residual = layers.Conv1D(out_channels, 1, padding='same', use_bias=True)(x)
    residual = layers.BatchNormalization()(residual)

    for i in range(R):
        x = tcs_conv_step(x, out_channels, kernel_size)
        x = layers.BatchNormalization()(x)
        
        if i == (R - 1):
            x = layers.Add()([x, residual])
            
        x = layers.ReLU()(x)
        x = layers.Dropout(0.1)(x)
        
    return x