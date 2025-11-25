from keras.api import layers, models, Input
from keras.api import regularizers 

def residual_block_type_one(input_shape, c, k, l2_rate=0.001):
    multiplied_size = int(c*k)
    x = layers.Conv1D(multiplied_size, 3, strides=1, use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_rate))(input_shape)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv1D(multiplied_size, 3, strides=1, use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_rate))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, input_shape])
    x = layers.ReLU()(x)
    return x

def residual_block_type_two(input_shape, c, k, l2_rate=0.001):
    multiplied_size = int(c*k)
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

def resnet8(input_shape, num_classes, k, l2_rate=0.001):
    input_layer = layers.Input(shape=input_shape)
    x = layers.Conv1D(int(16*k), 1, strides=1, use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_rate))(input_layer)
    x = residual_block_type_two(x, 24, k, l2_rate)
    x = residual_block_type_two(x, 32, k, l2_rate)
    x = residual_block_type_two(x, 48, k, l2_rate)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(num_classes, activation='sigmoid', kernel_regularizer=regularizers.l2(l2_rate))(x)
    model = models.Model(inputs=input_layer, outputs=output)
    return model

def resnet14(input_shape, num_classes, k, l2_rate=0.001):
    input_layer = layers.Input(shape=input_shape)
    x = layers.Conv1D(int(16 * k), 3, strides=1, use_bias=False, padding='same', kernel_regularizer=regularizers.l2(l2_rate))(input_layer)
    x = residual_block_type_two(x, 24, k, l2_rate)
    x = residual_block_type_one(x, 24, k, l2_rate)
    x = residual_block_type_two(x, 32, k, l2_rate)
    x = residual_block_type_one(x, 32, k, l2_rate)
    x = residual_block_type_two(x, 48, k, l2_rate)
    x = residual_block_type_one(x, 48, k, l2_rate)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(num_classes, activation='sigmoid', kernel_regularizer=regularizers.l2(l2_rate))(x)
    model = models.Model(inputs=input_layer, outputs=output)
    return model

def tcs_conv_block(x, out_channels, kernel_size, dropout=0.2):
    x = layers.SeparableConv1D(
        filters=out_channels,
        kernel_size=kernel_size,
        strides=1,
        padding='same',
        use_bias=False
    )(x)
    
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(dropout)(x)
    return x

def main_block(x, in_channels, out_channels, kernel_size, R=1):
    residual = layers.Conv1D(out_channels, 1, padding='same', use_bias=False)(x)
    residual = layers.BatchNormalization()(residual)

    for i in range(R):
        if i == 0:
            x = tcs_conv_block(x, out_channels, kernel_size)
        else:
            x = tcs_conv_block(x, out_channels, kernel_size)

    x = layers.Add()([x, residual])
    x = layers.ReLU()(x)
    return x

def MatchboxNet(input_shape=(98, 20), num_classes=1, B=3, R=1, C=64):
    inputs = Input(shape=input_shape)
    
    kernel_sizes = [k*2+11 for k in range(1, B+1)] 

    x = layers.Conv1D(128, 11, strides=2, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = main_block(x, 128, C, kernel_sizes[0], R=R)

    for i in range(1, B):
        x = main_block(x, C, C, kernel_sizes[i], R=R)

    x = layers.Conv1D(128, 29, dilation_rate=2, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv1D(128, 1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv1D(num_classes, 1, padding='same')(x)

    x = layers.GlobalAveragePooling1D()(x)

    if num_classes == 1:
        outputs = layers.Activation('sigmoid')(x)
    else:
        outputs = layers.Activation('softmax')(x)

    return models.Model(inputs=inputs, outputs=outputs)