from keras.api import layers, models, Input, regularizers

from models.blocks import (
    residual_block_type_one, 
    residual_block_type_two, 
    matchbox_main_block
)

def resnet8(input_shape, num_classes, k, l2_rate=0.001):
    input_layer = layers.Input(shape=input_shape)
    
    x = layers.Conv1D(
        int(16*k), 1, 
        strides=1, 
        use_bias=False,
        padding='same',
        kernel_regularizer=regularizers.l2(l2_rate) 
    )(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)  
    
    x = residual_block_type_two(x, 24, k, l2_rate)
    x = layers.Dropout(0.15)(x)
    
    x = residual_block_type_two(x, 32, k, l2_rate)
    x = layers.Dropout(0.2)(x)  
    
    x = residual_block_type_two(x, 48, k, l2_rate)
    x = layers.Dropout(0.25)(x) 
    
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Flatten()(x)
    
    x = layers.Dense(
        int(64 * k), 
        activation='relu',
        kernel_regularizer=regularizers.l2(l2_rate)
    )(x)
    x = layers.Dropout(0.4)(x)  
    
    if num_classes == 1:
        output = layers.Dense(num_classes, activation='sigmoid')(x)
    else:
        output = layers.Dense(num_classes, activation='softmax')(x)
        
    return models.Model(inputs=input_layer, outputs=output, name="ResNet8")

def resnet14(input_shape, num_classes, k, l2_rate=0.001):
    input_layer = layers.Input(shape=input_shape)
    
    x = layers.Conv1D(
        int(16 * k), 3, 
        strides=1, 
        use_bias=False, 
        padding='same',
        kernel_regularizer=regularizers.l2(l2_rate)
    )(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    
    x = residual_block_type_two(x, 24, k, l2_rate)
    x = layers.Dropout(0.15)(x) 
    
    x = residual_block_type_one(x, 24, k, l2_rate)
    x = layers.Dropout(0.15)(x) 
    
    x = residual_block_type_two(x, 32, k, l2_rate)
    x = layers.Dropout(0.2)(x)
    
    x = residual_block_type_one(x, 32, k, l2_rate)
    x = layers.Dropout(0.2)(x) 
    
    x = residual_block_type_two(x, 48, k, l2_rate)
    x = layers.Dropout(0.25)(x) 
    
    x = residual_block_type_one(x, 48, k, l2_rate)
    x = layers.Dropout(0.25)(x)
    
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Flatten()(x)
    
    x = layers.Dense(
        int(128 * k), 
        activation='relu',
        kernel_regularizer=regularizers.l2(l2_rate)
    )(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(
        int(64 * k), 
        activation='relu',
        kernel_regularizer=regularizers.l2(l2_rate)
    )(x)
    x = layers.Dropout(0.4)(x)
    
    if num_classes == 1:
        output = layers.Dense(num_classes, activation='sigmoid')(x)
    else:
        output = layers.Dense(num_classes, activation='softmax')(x)
        
    return models.Model(inputs=input_layer, outputs=output, name="ResNet14")

def crnn(input_shape, num_classes, k, l2_rate=0.001):
    input_layer = layers.Input(shape=input_shape)

    x = layers.Conv1D(
        filters=int(32 * k),
        kernel_size=3,
        strides=1,
        padding='causal',
        use_bias=False,
        kernel_regularizer=regularizers.l2(l2_rate)
    )(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.1)(x) 

    x = layers.GRU(
        units=int(64 * k),
        return_sequences=False,
        kernel_regularizer=regularizers.l2(l2_rate),
        recurrent_dropout=0.15 
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x) 

    x = layers.Dense(
        units=int(32 * k),
        activation='relu',
        kernel_regularizer=regularizers.l2(l2_rate)
    )(x)
    x = layers.Dropout(0.4)(x)

    if num_classes == 1:
        output_layer = layers.Dense(num_classes, activation='sigmoid')(x)
    else:
        output_layer = layers.Dense(num_classes, activation='softmax')(x)

    return models.Model(inputs=input_layer, outputs=output_layer, name="CRNN")

def MatchboxNet(input_shape=(98, 20), num_classes=1, B=3, R=1, C=64):
    inputs = Input(shape=input_shape)
    
    x = layers.GroupNormalization(groups=-1)(inputs)

    kernel_sizes = [k*2+11 for k in range(1, B+1)] 

    x = layers.Conv1D(128, 11, strides=2, padding='same', use_bias=True)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = matchbox_main_block(x, C, kernel_sizes[0], R=R)
    
    for i in range(1, B):
        x = matchbox_main_block(x, C, kernel_sizes[i], R=R)

    x = layers.Conv1D(128, 29, dilation_rate=2, padding='same', use_bias=True)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv1D(128, 1, padding='same', use_bias=True)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv1D(num_classes, 1, padding='same', use_bias=True)(x)
    
    x = layers.GlobalAveragePooling1D()(x)

    if num_classes == 1:
        outputs = layers.Activation('sigmoid')(x)
    else:
        outputs = layers.Activation('softmax')(x)

    return models.Model(inputs=inputs, outputs=outputs, name="MatchboxNet")